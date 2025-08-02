import time
from copy import deepcopy
import torch
import numpy as np
from statsmodels.stats.correlation_tools import cov_nearest
from servers.server_base import Server
from clients.client_fedfda import ClientFedFDA
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score

class ServerFedFDA(Server):
    def __init__(self, args):
        super().__init__(args)
        self.clients = [
            ClientFedFDA(args, i) for i in range(self.num_clients)
        ]
        self.global_means = torch.Tensor(torch.rand([self.num_classes, self.D]))
        self.global_covariance = torch.Tensor(torch.eye(self.D))
        self.global_priors = torch.ones(self.num_classes) / self.num_classes
        self.r = 0

    def train(self):
        for r in range(1, self.global_rounds+1):
            start_time = time.time()
            self.r = r
            if r == (self.global_rounds): # 最后一轮全员参与
                self.sampling_prob = 1.0
            self.sample_active_clients()
            self.send_models()

            # 训练客户端
            train_acc, train_loss = self.train_clients()
            train_time = time.time() - start_time
            
            # 聚合客户端模型
            self.aggregate_models()
            
            round_time = time.time() - start_time
            self.train_times.append(train_time)
            self.round_times.append(round_time)

            # 日志记录
            if r % self.eval_gap == 0 or r == self.global_rounds:
                print(f"--- Round [{r}/{self.global_rounds}] Evaluation ---")
                ptest_acc, ptest_loss, ptest_acc_std = self.evaluate_personalized()  
                print(f"  [Main] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  [Main] Test Loss: {ptest_loss:.4f}, Test Acc: {ptest_acc:.2f}% (std: {ptest_acc_std:.2f})")
                print(f"  [Time] Train: {train_time:.2f}s, Round: {round_time:.2f}s")
            else:
                print(f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}]\t Train Time [{train_time:.2f}]")

    def aggregate_models(self):
        # 聚合基础模型
        super().aggregate_models()

        # 聚合高斯估计
        total_samples = sum(c.num_train for c in self.active_clients)
        if total_samples == 0: return

        self.global_means.data = torch.zeros_like(self.clients[0].means)
        self.global_covariance.data = torch.zeros_like(self.clients[0].covariance)
        
        for c in self.active_clients:
            self.global_means.data += (c.num_train / total_samples) * c.adaptive_means.data
            self.global_covariance.data += (c.num_train / total_samples) * c.adaptive_covariance.data
    
    def send_models(self):
        # 发送基础模型
        super().send_models()
        # 发送全局高斯估计
        for c in self.active_clients:
            c.global_means.data = self.global_means.data
            c.global_covariance.data = self.global_covariance.data
            if self.r == 1:
                c.means.data = self.global_means.data
                c.covariance.data = self.global_covariance.data
                c.adaptive_means.data = self.global_means.data
                c.adaptive_covariance.data = self.global_covariance.data

    def _calculate_ece(self, confidences, predictions, labels, n_bins=15):
        """计算期望校准误差 (ECE)"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = (predictions == labels)
        ece = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    def _calculate_brier_score(self, probs, labels):
        """计算Brier分数"""
        one_hot_labels = np.eye(self.num_classes)[labels]
        return np.mean(np.sum((probs - one_hot_labels)**2, axis=1))

    def _calculate_fpr_at_tpr(self, id_scores, ood_scores, tpr_threshold=0.95):
        """在给定的TPR阈值下计算FPR"""
        labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        # 分数越高表示越可能为OOD，因此能量分数是合适的
        scores = np.concatenate([id_scores, ood_scores])
        fpr, tpr, _ = roc_curve(labels, scores)
        
        if np.all(tpr < tpr_threshold):
            return 1.0
        # 找到第一个tpr >= tpr_threshold的位置
        idx = np.searchsorted(tpr, tpr_threshold, side='left')
        if idx == len(tpr): # 如果所有tpr都小于阈值
            return 1.0
        return fpr[idx]

    def evaluate_personalized(self):
        """
        更新客户端的beta，计算本地插值统计量(mu, Sigma)，
        并使用插值统计量和本地先验评估模型性能，同时计算新的指标。
        """
        total_test_samples = sum(c.num_test for c in self.clients if c.num_test > 0)
        if total_test_samples == 0:
            print("所有客户端均无测试样本，跳过评估。")
            return 0.0, 0.0, 0.0

        weighted_loss, weighted_acc = 0.0, 0.0
        accs, kl_divs = [], []
        
        all_id_probs, all_id_labels = [], []
        all_id_uncertainties, all_ood_uncertainties = [], []

        for c in self.clients:
            if c.num_test == 0: continue
            
            old_model = deepcopy(c.model)
            c.model = deepcopy(self.model)
            c.global_means.data = self.global_means.data
            c.global_covariance.data = self.global_covariance.data
            c.global_means = c.global_means.to(self.device)
            c.global_covariance = c.global_covariance.to(self.device)
            c.model.eval()

            # 个性化客户端模型
            c_feats, c_labels = c.compute_feats(split="train")
            c.solve_beta(feats=c_feats, labels=c_labels)
            means_mle, scatter_mle, _, counts = c.compute_mle_statistics(feats=c_feats, labels=c_labels)
            means_mle_filled = torch.stack([means_mle[i] if means_mle[i] is not None and counts[i] > c.min_samples else c.global_means[i] for i in range(self.num_classes)])
            cov_mle = (scatter_mle / (np.sum(counts)-1)) + 1e-4 + torch.eye(c.D).to(c.device)
            cov_psd = torch.Tensor(cov_nearest(cov_mle.cpu().numpy(), method="clipped")).to(c.device)
            c.update(means_mle_filled, cov_psd)
            c.set_lda_weights(c.adaptive_means, c.adaptive_covariance)

            # --- ID 数据评估 ---
            id_logits, id_labels, id_loss = c.get_logits_labels_and_loss(split="test")
            
            if id_logits.numel() > 0:
                id_probs = F.softmax(id_logits, dim=1)
                preds = torch.argmax(id_probs, dim=1)
                acc = (preds == id_labels).float().mean() * 100.0
                
                accs.append(acc.cpu())
                weighted_loss += (c.num_test / total_test_samples) * id_loss
                weighted_acc += (c.num_test / total_test_samples) * acc.item()
                
                all_id_probs.append(id_probs.numpy())
                all_id_labels.append(id_labels.numpy())
                
                # 不确定性度量: 能量分数 (越高越不确定)
                id_energy = -torch.logsumexp(id_logits, dim=1)
                all_id_uncertainties.append(id_energy.numpy())

            # --- OOD 数据评估 ---
            ood_logits, _, _ = c.get_logits_labels_and_loss(split="ood")
            if ood_logits.numel() > 0:
                ood_energy = -torch.logsumexp(ood_logits, dim=1)
                all_ood_uncertainties.append(ood_energy.numpy())
            
            # 恢复模型并清理
            c.model = old_model
            c.model.to("cpu")
            c.global_means.to("cpu")
            c.global_covariance.to("cpu")
            c.adaptive_means.to("cpu")
            c.adaptive_covariance.to("cpu")
            c.means.to("cpu")
            c.covariance.to("cpu")

        std = torch.std(torch.stack(accs)) if accs else torch.tensor(0.0)

        # --- 计算并打印高级指标 ---
        if all_id_probs:
            id_probs_np = np.concatenate(all_id_probs)
            id_labels_np = np.concatenate(all_id_labels)
            
            brier_score = self._calculate_brier_score(id_probs_np, id_labels_np)
            
            confidences = np.max(id_probs_np, axis=1)
            predictions = np.argmax(id_probs_np, axis=1)
            ece = self._calculate_ece(confidences, predictions, id_labels_np)
            
            print(f"  [Calibration] Brier: {brier_score:.4f}, ECE: {ece:.4f}")

        if all_id_uncertainties and all_ood_uncertainties:
            id_uncertainties_np = np.concatenate(all_id_uncertainties)
            ood_uncertainties_np = np.concatenate(all_ood_uncertainties)
            
            labels = np.concatenate([np.zeros_like(id_uncertainties_np), np.ones_like(ood_uncertainties_np)])
            scores = np.concatenate([id_uncertainties_np, ood_uncertainties_np])
            auroc = roc_auc_score(labels, scores)
            
            fpr95 = self._calculate_fpr_at_tpr(id_uncertainties_np, ood_uncertainties_np)
            
            print(f"  [OOD Detection] AUROC: {auroc:.4f}, FPR@95TPR: {fpr95:.4f}")
        
        return weighted_acc, weighted_loss, std