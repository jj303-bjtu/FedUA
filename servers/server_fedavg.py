import time
from copy import deepcopy
import torch
import numpy as np
from servers.server_base import Server
from clients.client_fedavg import ClientFedAvg
from sklearn.metrics import roc_auc_score, roc_curve

class ServerFedAvg(Server):
    def __init__(self, args):
        super().__init__(args)
        self.clients = []
        for client_idx in range(self.num_clients):
            # The existing client is sufficient
            c = ClientFedAvg(args, client_idx)
            self.clients.append(c)

    def send_models(self):
        for c in self.active_clients:
            c.set_model(self.model)

    def _calculate_ece(self, probs, labels, n_bins=15):
        """计算期望校准误差 (ECE)"""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels)

        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece

    def _calculate_fpr_at_tpr(self, id_values, ood_values, tpr_threshold=0.95):
        """在给定的TPR阈值下计算FPR"""
        if len(id_values) == 0 or len(ood_values) == 0:
            return -1.0

        y_true = np.concatenate([np.zeros(len(id_values)), np.ones(len(ood_values))])
        y_score = np.concatenate([id_values, ood_values])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        
        if np.sum(tpr >= tpr_threshold) == 0:
            return 1.0 # 如果无法达到阈值，则返回最坏情况
            
        idx = np.where(tpr >= tpr_threshold)[0][0]
        return fpr[idx]

    def evaluate(self):
        """使用高级指标评估 **全局模型**"""
        all_id_probs, all_id_labels, all_id_uncertainties, all_ood_uncertainties = [], [], [], []
        
        for c in self.clients:
            # Temporarily set the client's model to the global model for evaluation
            original_model = deepcopy(c.model)
            c.model = deepcopy(self.model)
            
            # Get raw outputs for metric calculation
            eval_results = c.get_eval_output()
            
            if eval_results['id_labels'].size > 0:
                all_id_probs.append(eval_results['id_probs'])
                all_id_labels.append(eval_results['id_labels'])
                all_id_uncertainties.append(eval_results['id_uncertainties'])

            if eval_results['ood_uncertainties'].size > 0:
                all_ood_uncertainties.append(eval_results['ood_uncertainties'])
            
            # Restore the client's original model
            c.model = original_model

        if not all_id_labels:
            return 0, 0, 0, 0, 0, 0
        
        # Aggregate results from all clients
        all_id_probs = np.concatenate(all_id_probs)
        all_id_labels = np.concatenate(all_id_labels)
        all_id_uncertainties = np.concatenate(all_id_uncertainties)
        all_ood_uncertainties = np.concatenate(all_ood_uncertainties)

        # --- Calculate all metrics on aggregated results ---
        predictions = np.argmax(all_id_probs, axis=1)
        acc = np.mean(predictions == all_id_labels) * 100.0
        log_probs = -np.log(all_id_probs[range(len(all_id_labels)), all_id_labels.astype(int)])
        loss = np.mean(log_probs)

        num_classes = all_id_probs.shape[1]
        id_labels_one_hot = np.eye(num_classes)[all_id_labels.astype(int)]
        brier_score = np.mean(np.sum((all_id_probs - id_labels_one_hot)**2, axis=1))
        
        ece = self._calculate_ece(all_id_probs, all_id_labels)

        auroc, fpr_at_95_tpr = -1.0, -1.0
        if all_ood_uncertainties.size > 0 and all_id_uncertainties.size > 0:
            y_true = np.concatenate([np.zeros(len(all_id_uncertainties)), np.ones(len(all_ood_uncertainties))])
            y_score = np.concatenate([all_id_uncertainties, all_ood_uncertainties])
            auroc = roc_auc_score(y_true, y_score)
            fpr_at_95_tpr = self._calculate_fpr_at_tpr(all_id_uncertainties, all_ood_uncertainties)

        return acc, loss, brier_score, ece, auroc, fpr_at_95_tpr

    # =================================================================================
    # >>> NEW FUNCTION FOR FEDAVG-FT <<<
    # =================================================================================
    def evaluate_fine_tuned(self):
        """
        在每个客户端上微调全局模型，然后评估生成的个性化模型。
        """
        print("\n--- FedAvgFT: Starting Fine-tuning and Personalized Evaluation ---")
        all_id_probs, all_id_labels, all_id_uncertainties, all_ood_uncertainties = [], [], [], []
        
        # Iterate over ALL clients to create personalized models
        for c in self.clients:
            # 1. Send the final global model to the client
            c.set_model(self.model)
            
            # 2. Fine-tune the model on local data. The client's `train()` method does this.
            #    The fine-tuned model is now stored in `c.model`.
            c.train() 
            
            # 3. Evaluate the now-personalized model using the existing get_eval_output function.
            #    This function will use the fine-tuned `c.model`.
            eval_results = c.get_eval_output()
            
            # 4. Aggregate the raw results from this client's personalized model
            if eval_results['id_labels'].size > 0:
                all_id_probs.append(eval_results['id_probs'])
                all_id_labels.append(eval_results['id_labels'])
                all_id_uncertainties.append(eval_results['id_uncertainties'])

            if eval_results['ood_uncertainties'].size > 0:
                all_ood_uncertainties.append(eval_results['ood_uncertainties'])

        if not all_id_labels:
            print("FedAvgFT: No evaluation data found for personalized models.")
            return
        
        # 5. Aggregate results from ALL personalized models
        all_id_probs = np.concatenate(all_id_probs)
        all_id_labels = np.concatenate(all_id_labels)
        all_id_uncertainties = np.concatenate(all_id_uncertainties)
        all_ood_uncertainties = np.concatenate(all_ood_uncertainties)

        # 6. Calculate the same set of metrics on the aggregated personalized results
        predictions = np.argmax(all_id_probs, axis=1)
        acc = np.mean(predictions == all_id_labels) * 100.0
        log_probs = -np.log(all_id_probs[range(len(all_id_labels)), all_id_labels.astype(int)])
        loss = np.mean(log_probs)

        num_classes = all_id_probs.shape[1]
        id_labels_one_hot = np.eye(num_classes)[all_id_labels.astype(int)]
        brier_score = np.mean(np.sum((all_id_probs - id_labels_one_hot)**2, axis=1))
        
        ece = self._calculate_ece(all_id_probs, all_id_labels)

        auroc, fpr_at_95_tpr = -1.0, -1.0
        if all_ood_uncertainties.size > 0 and all_id_uncertainties.size > 0:
            y_true = np.concatenate([np.zeros(len(all_id_uncertainties)), np.ones(len(all_ood_uncertainties))])
            y_score = np.concatenate([all_id_uncertainties, all_ood_uncertainties])
            auroc = roc_auc_score(y_true, y_score)
            fpr_at_95_tpr = self._calculate_fpr_at_tpr(all_id_uncertainties, all_ood_uncertainties)

        # 7. Print the final FedAvgFT metrics
        print(f"--- FedAvgFT Personalized Metrics ---")
        print(f"  Test Loss:  {loss:.4f}\t Test Acc:  {acc:.2f}%")
        print(f"  Calibration: Brier={brier_score:.4f}, ECE={ece:.4f}")
        if auroc != -1.0:
            print(f"  OOD Detection: AUROC={auroc:.4f}, FPR@95TPR={fpr_at_95_tpr:.4f}")

    # =================================================================================
    # >>> MODIFIED MAIN TRAINING LOOP <<<
    # =================================================================================
    def train(self):
        # --- Standard FedAvg Training Phase ---
        for r in range(1, self.global_rounds+1):
            start_time = time.time()
            self.sample_active_clients()
            self.send_models()

            train_acc, train_loss = self.train_clients()
            self.aggregate_models()
            
            round_time = time.time() - start_time
            self.round_times.append(round_time)

            if r % self.eval_gap == 0 or r == self.global_rounds:
                # This evaluates the GLOBAL model during training
                test_acc, test_loss, brier, ece, auroc, fpr95tpr = self.evaluate()
                
                print(f"--- [Global Model] Round [{r}/{self.global_rounds}] ---")
                print(f"  Train Loss: {train_loss:.4f}\t Train Acc: {train_acc:.2f}%")
                print(f"  Test Loss:  {test_loss:.4f}\t Test Acc:  {test_acc:.2f}%")
                print(f"  Calibration: Brier={brier:.4f}, ECE={ece:.4f}")
                if auroc != -1.0:
                    print(f"  OOD Detection: AUROC={auroc:.4f}, FPR@95TPR={fpr95tpr:.4f}")
                print(f"  Time: {round_time:.2f}s")
            else:
                print(f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}%]\t Time [{round_time:.2f}s]")

        # --- FedAvgFT Evaluation Phase (runs once after global training) ---
        self.evaluate_fine_tuned()

    def train_clients(self):
        """Helper function to train active clients and return average metrics."""
        total_samples = 0
        weighted_acc = 0
        weighted_loss = 0
        for c in self.active_clients:
            acc, loss = c.train()
            num_samples = c.num_train
            weighted_acc += acc * num_samples
            weighted_loss += loss * num_samples
            total_samples += num_samples
        return weighted_acc / total_samples, weighted_loss / total_samples

    def aggregate_models(self):
        """Aggregates models from active clients."""
        total_samples = sum(c.num_train for c in self.active_clients)
        if total_samples == 0:
            return

        global_state_dict = self.model.state_dict()
        aggregated_state_dict = {name: torch.zeros_like(param) for name, param in global_state_dict.items()}

        for c in self.active_clients:
            weight = c.num_train / total_samples
            client_state_dict = c.model.state_dict()
            for name, param in client_state_dict.items():
                if param.dtype.is_floating_point:
                    aggregated_state_dict[name] += param.data * weight
                else:
                    aggregated_state_dict[name].copy_(param.data)
        
        self.model.load_state_dict(aggregated_state_dict)