import torch
import numpy as np
import time
import os # ### SUGGESTION 1: Added for saving plots
from copy import deepcopy
from servers.server_base import Server
from clients.client_fedug_v2 import ClientFedUgV2
from utils.util import AverageMeter
### SUGGESTION 1: UNCERTAINTY QUANTIFICATION ###
from utils.uncertainty_metrics import calculate_brier_score, calculate_ece, plot_reliability_diagram, calculate_ood_auc
from sklearn.metrics import average_precision_score # 用于 AUC-PR
from scipy.stats import spearmanr # 用于认知不确定性保真度
from scipy.stats import ttest_ind # 用于偶然不确定性一致性
import matplotlib.pyplot as plt

class ServerFedUgV2(Server):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.clients = [ClientFedUgV2(args, i) for i in range(args.num_clients)]
        
        self.global_prior_alpha = torch.ones(self.num_classes) * 2.0
        
        self.round_times = []
        self.train_times = []

        ### SUGGESTION 1: Create directory for saving evaluation results
        self.results_dir = f"results/{args.dataset}_{args.method}_{args.dir_alpha}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Initialized FedUG V2 Server with {len(self.clients)} clients")
    
    def send_models(self):
        for client in self.clients:
            client.model.load_state_dict(self.model.state_dict())
            client.model_global.load_state_dict(self.model.state_dict())
            client.set_global_prior_alpha(self.global_prior_alpha)
    
    def train_round(self, round_num):
        self.logger.info(f"--- Global Round {round_num}/{self.global_rounds} ---")
        self.sample_active_clients()
        active_clients = self.active_clients
        
        self.send_models()
        
        client_updates = []
        client_alpha_reports = []
        client_data_sizes = []
        
        kl_div_times = AverageMeter()

        for client in active_clients:
            client.set_current_global_round(round_num)
            stats = client.train(round_num)
            
            kl_div_times.update(stats.get('kl_div_time', 0))

            model_weights, alpha_report, data_size = client.get_update()
            
            client_updates.append(model_weights)
            client_alpha_reports.append(alpha_report)
            client_data_sizes.append(data_size)

        # 报告开销分析
        avg_kl_time = kl_div_times.avg * 1000
        self.logger.info(f"[Overhead Analysis] Avg. KL-Div computation time per batch: {avg_kl_time:.4f} ms")
        if client_alpha_reports:
            report_size_bytes = client_alpha_reports[0].element_size() * client_alpha_reports[0].nelement()
            self.logger.info(f"[Overhead Analysis] Communication overhead per client for alpha_report: {report_size_bytes} bytes")

        if client_updates:
            self.aggregate_models(client_updates, client_data_sizes)
            
            # 使用拜占庭鲁棒的先验聚合替代原来的方法
            if hasattr(self.args, 'use_byzantine_robust') and self.args.use_byzantine_robust:
                self.logger.info("Using Byzantine-robust prior aggregation")
                self.aggregate_alpha_priors(active_clients)
            else:
                # 原来的聚合方法
                self.aggregate_alpha_reports(client_alpha_reports, client_data_sizes)
        
        return 0.0, 0.0
    
    def aggregate_models(self, client_updates, client_data_sizes):
        if not client_updates: return
        total_data_size = sum(client_data_sizes)
        if total_data_size == 0: return
        
        global_state_dict = self.model.state_dict()
        aggregated_state_dict = {name: torch.zeros_like(param) for name, param in global_state_dict.items()}
        
        for i, client_state_dict in enumerate(client_updates):
            weight = client_data_sizes[i] / total_data_size
            for name in aggregated_state_dict:
                if 'num_batches_tracked' not in name:
                    aggregated_state_dict[name] += weight * client_state_dict[name]
        
        self.model.load_state_dict(aggregated_state_dict)
    
    def aggregate_alpha_reports(self, alpha_reports, client_data_sizes):
        """聚合客户端的alpha报告"""
        if not alpha_reports: return
        
        total_size = sum(client_data_sizes)
        if total_size == 0: return
        
        weighted_sum = torch.zeros_like(alpha_reports[0])
        for report, size in zip(alpha_reports, client_data_sizes):
            weighted_sum += (size / total_size) * report.to(weighted_sum.device)
        
        ### SUGGESTION 2: ABLATION STUDY ###
        # 条件应用平滑和限制
        if hasattr(self.args, 'use_prior_smoothing') and self.args.use_prior_smoothing:
            smoothing_factor = 0.1
            uniform_prior = torch.ones_like(weighted_sum) * 2.0
            aggregated_alpha = (1 - smoothing_factor) * weighted_sum + smoothing_factor * uniform_prior
        else:
            aggregated_alpha = weighted_sum
        
        if hasattr(self.args, 'use_prior_clamping') and self.args.use_prior_clamping:
            self.global_prior_alpha = torch.clamp(aggregated_alpha, min=1.1, max=5.0)
        else:
            self.global_prior_alpha = aggregated_alpha
        
        # 记录alpha值统计
        self.logger.info(f"Global Prior Alpha Stats: Min={self.global_prior_alpha.min().item():.2f}, "
                        f"Max={self.global_prior_alpha.max().item():.2f}, "
                        f"Mean={self.global_prior_alpha.mean().item():.2f}")

    def evaluate_uncertainty(self, round_idx):
        """评估不确定性性能"""
        self.logger.info(f"--- Evaluating models at round {round_idx} with Advanced Uncertainty Metrics ---")
        
        # 评估指标
        personal_accs = []
        personal_briers = []
        personal_eces = []
        epistemic_fidelities = []
        aleatoric_consistencies = []
        
        # 收集OOD评估数据
        all_id_uncertainties = []
        all_ood_uncertainties = []
        all_id_energies = []
        all_ood_energies = []
        all_id_total = []
        all_ood_total = []
        
        # 评估每个客户端
        for client in self.clients:
            # ID数据评估
            id_results = client.get_eval_output(use_personal=True, dataset='test')
            if id_results is None:
                continue
                
            # 计算准确率
            preds = np.argmax(id_results['probs'], axis=1)
            acc = np.mean(preds == id_results['labels']) * 100
            personal_accs.append(acc)
            
            # 计算Brier分数
            if 'brier_score' in id_results:
                personal_briers.append(id_results['brier_score'])
            else:
                y_one_hot = np.zeros((len(id_results['labels']), id_results['probs'].shape[1]))
                for i, label in enumerate(id_results['labels']):
                    y_one_hot[i, label] = 1
                brier = np.mean(np.sum((id_results['probs'] - y_one_hot) ** 2, axis=1))
                personal_briers.append(brier)
            
            # 计算ECE
            if 'ece' in id_results:
                personal_eces.append(id_results['ece'])
            else:
                confidences = np.max(id_results['probs'], axis=1)
                ece = calculate_ece(confidences, preds == id_results['labels'])
                personal_eces.append(ece)
            
            # 计算认知不确定性与错误的相关性
            is_correct = (preds == id_results['labels']).astype(float)
            if len(np.unique(is_correct)) > 1:  # 确保有正确和错误的预测
                epistemic_corr, _ = spearmanr(id_results['epistemic_uncertainties'], 1 - is_correct)
                if not np.isnan(epistemic_corr):
                    epistemic_fidelities.append(epistemic_corr)
            
            # 计算偶然不确定性一致性
            if 'aleatoric_uncertainties' in id_results:
                correct_mask = is_correct == 1
                incorrect_mask = is_correct == 0
                
                if np.sum(correct_mask) > 0 and np.sum(incorrect_mask) > 0:
                    # 使用t检验比较正确和错误预测的偶然不确定性
                    t_stat, p_value = ttest_ind(
                        id_results['aleatoric_uncertainties'][correct_mask],
                        id_results['aleatoric_uncertainties'][incorrect_mask],
                        equal_var=False
                    )
                    # 一致性指标：p值越小，差异越显著
                    aleatoric_consistencies.append(1 - min(p_value, 1.0))
            
            # 收集ID不确定性
            all_id_uncertainties.extend(id_results['epistemic_uncertainties'])
            if 'energy_uncertainties' in id_results:
                all_id_energies.extend(id_results['energy_uncertainties'])
            if 'total_uncertainties' in id_results:
                all_id_total.extend(id_results['total_uncertainties'])
            
            # OOD数据评估
            ood_results = client.get_eval_output(use_personal=True, dataset='ood')
            if ood_results is not None:
                all_ood_uncertainties.extend(ood_results['epistemic_uncertainties'])
                if 'energy_uncertainties' in ood_results:
                    all_ood_energies.extend(ood_results['energy_uncertainties'])
                if 'total_uncertainties' in ood_results:
                    all_ood_total.extend(ood_results['total_uncertainties'])
        
        # 计算平均指标
        mean_acc = 0
        if personal_accs:
            mean_acc = np.mean(personal_accs)
            self.logger.info(f"[Metric 1] Personalized Accuracy | Mean: {mean_acc:.2f}%, Std: {np.std(personal_accs):.2f}, Min: {np.min(personal_accs):.2f}, Max: {np.max(personal_accs):.2f}")
            self.logger.info(f"[Metric 1] Personalized Brier Score | Mean: {np.mean(personal_briers):.4f}, Std: {np.std(personal_briers):.4f}")
            self.logger.info(f"[Metric 1] Personalized ECE | Mean: {np.mean(personal_eces):.4f}, Std: {np.std(personal_eces):.4f}")
        
        if epistemic_fidelities:
            self.logger.info(f"[Metric 2] Epistemic Fidelity (Spearman Corr) | Mean: {np.mean(epistemic_fidelities):.4f}, Std: {np.std(epistemic_fidelities):.4f}")
        
        if aleatoric_consistencies:
            self.logger.info(f"[Metric 3] Aleatoric Consistency (U) | Mean: {np.mean(aleatoric_consistencies):.4f}, Std: {np.std(aleatoric_consistencies):.4f}")
        
        # OOD检测性能
        if all_id_uncertainties and all_ood_uncertainties:
            # 认知不确定性的OOD检测
            epistemic_auroc = calculate_ood_auc(all_id_uncertainties, all_ood_uncertainties)
            self.logger.info(f"[OOD Eval] OOD AUC-ROC (Epistemic): {epistemic_auroc:.4f}")
            
            # FPR@95%TPR
            fpr95 = self._calculate_fpr_at_tpr(all_id_uncertainties, all_ood_uncertainties, tpr_threshold=0.95)
            self.logger.info(f"[Metric 4] OOD FPR@95%TPR: {fpr95:.4f}")
            
            # AUPRC
            auprc = self._calculate_auprc(all_id_uncertainties, all_ood_uncertainties)
            self.logger.info(f"[Metric 5] OOD AUC-PR: {auprc:.4f}")
            
            # 能量不确定性的OOD检测
            if all_id_energies and all_ood_energies:
                energy_auroc = calculate_ood_auc(all_id_energies, all_ood_energies)
                self.logger.info(f"[OOD Eval] OOD AUC-ROC (Energy): {energy_auroc:.4f}")
            
            # 总不确定性的OOD检测
            if all_id_total and all_ood_total:
                total_auroc = calculate_ood_auc(all_id_total, all_ood_total)
                self.logger.info(f"[OOD Eval] OOD AUC-ROC (Total): {total_auroc:.4f}")
            
            # 绘制不确定性分布图
            self.plot_uncertainty_distributions(
                all_id_uncertainties, 
                all_id_total[:len(all_id_uncertainties)] if all_id_total else None,
                all_ood_uncertainties, 
                all_ood_total[:len(all_ood_uncertainties)] if all_ood_total else None,
                round_idx, 
                self.results_dir
            )
        
        # 返回平均准确率，用于保存最佳模型
        return mean_acc

    def train(self):
        self.logger.info("Starting FedUG V2 Training Process")
        best_acc = 0.0
        
        for round_num in range(1, self.global_rounds + 1):
            start_time = time.time()
            self.train_round(round_num)
            round_time = time.time() - start_time
            self.round_times.append(round_time)
            
            if round_num % self.args.eval_gap == 0 or round_num == self.global_rounds:
                # 使用evaluate_uncertainty方法替代evaluate_models
                test_acc = self.evaluate_uncertainty(round_num)
                if test_acc > best_acc:
                    best_acc = test_acc
                    self.logger.info(f"New best personalized accuracy: {best_acc:.2f}%")
                    
                    # 保存最佳模型
                    if hasattr(self.args, 'save_models') and self.args.save_models:
                        self._save_best_model(round_num, best_acc)
        
        self.logger.info(f"Training completed. Best personalized accuracy: {best_acc:.2f}%")
        return best_acc

    def _save_best_model(self, round_num, acc):
        """保存最佳模型"""
        save_path = os.path.join(self.results_dir, f"best_model_round{round_num}_acc{acc:.2f}.pt")
        torch.save({
            'model': self.model.state_dict(),
            'global_prior_alpha': self.global_prior_alpha,
            'round': round_num,
            'acc': acc
        }, save_path)
        self.logger.info(f"Saved best model to {save_path}")

    def plot_uncertainty_distributions(self, id_epistemic, id_aleatoric, ood_epistemic, ood_aleatoric, round_num, save_dir):
        """绘制不确定性分布图"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(id_epistemic, bins=50, alpha=0.7, label='ID Epistemic', density=True)
        if ood_epistemic is not None and len(ood_epistemic) > 0:
            plt.hist(ood_epistemic, bins=50, alpha=0.7, label='OOD Epistemic', density=True)
        plt.title(f'Epistemic Uncertainty (Round {round_num})')
        plt.xlabel('Uncertainty')
        plt.ylabel('Density')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(id_aleatoric, bins=50, alpha=0.7, label='ID Total', density=True)
        if ood_aleatoric is not None and len(ood_aleatoric) > 0:
            plt.hist(ood_aleatoric, bins=50, alpha=0.7, label='OOD Total', density=True)
        plt.title(f'Total Uncertainty (Round {round_num})')
        plt.xlabel('Uncertainty')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"uncertainty_dist_r{round_num}.png"))
        plt.close()

    def plot_error_vs_uncertainty(self, epistemic, aleatoric, errors, round_num, save_dir):
        """绘制错误与不确定性关系图"""
        correct_mask = ~errors
        
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(epistemic[correct_mask], aleatoric[correct_mask], alpha=0.1, s=5, label='Correct')
        plt.scatter(epistemic[errors], aleatoric[errors], alpha=0.1, s=5, label='Incorrect')
        plt.xlabel('Epistemic Uncertainty')
        plt.ylabel('Total Uncertainty')
        plt.title(f'Uncertainty of Predictions (Round {round_num})')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')

        plt.subplot(1, 2, 2)
        plt.boxplot([epistemic[correct_mask], epistemic[errors]], labels=['Correct', 'Incorrect'])
        plt.title('Epistemic Uncertainty vs. Error')
        plt.ylabel('Epistemic Uncertainty')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"error_vs_uncertainty_r{round_num}.png"))
        plt.close()

    def _expected_calibration_error(self, confidences, accuracies, num_bins=10):
        """计算期望校准误差 (ECE)"""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 计算落在这个bin中的样本
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    def _spearman_correlation(self, x, y):
        """计算Spearman相关系数"""
        from scipy.stats import spearmanr
        corr, _ = spearmanr(x, y)
        return corr if not np.isnan(corr) else 0.0

    def _calculate_auroc(self, id_values, ood_values):
        """计算AUC-ROC"""
        from sklearn.metrics import roc_auc_score
        # 确保ID为负类(0)，OOD为正类(1)
        y_true = np.concatenate([np.zeros(len(id_values)), np.ones(len(ood_values))])
        y_score = np.concatenate([id_values, ood_values])
        return roc_auc_score(y_true, y_score)

    def _calculate_fpr_at_tpr(self, id_values, ood_values, tpr_threshold=0.95):
        """计算在给定TPR阈值下的FPR"""
        from sklearn.metrics import roc_curve
        # 确保ID为负类(0)，OOD为正类(1)
        y_true = np.concatenate([np.zeros(len(id_values)), np.ones(len(ood_values))])
        y_score = np.concatenate([id_values, ood_values])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        
        # 找到最接近目标TPR的索引
        idx = np.argmin(np.abs(tpr - tpr_threshold))
        return fpr[idx]

    def _calculate_auprc(self, id_values, ood_values):
        """计算AUC-PR曲线下面积"""
        from sklearn.metrics import average_precision_score
        # 确保ID为负类(0)，OOD为正类(1)
        y_true = np.concatenate([np.zeros(len(id_values)), np.ones(len(ood_values))])
        y_score = np.concatenate([id_values, ood_values])
        return average_precision_score(y_true, y_score)

    def aggregate_alpha_priors(self, selected_clients):
        """拜占庭鲁棒的先验聚合"""
        alpha_priors = []
        client_weights = []
        
        for client in selected_clients:
            # 使用增强的get_alpha_prior方法
            alpha_prior = client.get_alpha_prior(None)  # 让客户端自己加载数据
            alpha_priors.append(alpha_prior)
            
            # 计算客户端权重（基于数据质量）
            data_size = client.num_train if hasattr(client, 'num_train') else 100
            
            # 检测异常先验
            alpha_sum = alpha_prior.sum().item()
            alpha_var = alpha_prior.var().item()
            alpha_max = alpha_prior.max().item()
            alpha_min = alpha_prior.min().item()
            
            # 增强的异常检测
            is_anomaly = (alpha_sum > 20.0 or alpha_var > 5.0 or 
                         alpha_max > 10.0 or alpha_min < 0.5 or
                         (alpha_max / (alpha_min + 1e-6)) > 20.0)
            
            if is_anomaly:
                weight = data_size * 0.05  # 进一步降低异常客户端权重
                self.logger.warning(f"Client {client.client_idx}: Anomalous prior detected "
                                  f"(sum={alpha_sum:.2f}, var={alpha_var:.2f}, max/min={alpha_max/alpha_min:.2f})")
            else:
                weight = data_size
            
            client_weights.append(weight)
        
        if not alpha_priors:
            self.global_prior_alpha = torch.ones(self.num_classes) * 1.5
            return
        
        # 使用加权中位数聚合（更鲁棒）
        alpha_stack = torch.stack(alpha_priors)
        weights_tensor = torch.tensor(client_weights, dtype=torch.float32)
        
        if weights_tensor.sum() == 0:
            self.global_prior_alpha = torch.ones(self.num_classes) * 1.5
            return
        
        weights_tensor = weights_tensor / weights_tensor.sum()
        
        # 计算加权中位数
        weighted_alpha = torch.zeros_like(alpha_priors[0])
        for i in range(self.num_classes):
            class_alphas = alpha_stack[:, i]
            # 使用加权中位数
            sorted_indices = torch.argsort(class_alphas)
            sorted_weights = weights_tensor[sorted_indices]
            cumsum_weights = torch.cumsum(sorted_weights, dim=0)
            median_idx = torch.searchsorted(cumsum_weights, 0.5)
            median_idx = min(median_idx.item(), len(class_alphas) - 1)
            weighted_alpha[i] = class_alphas[sorted_indices[median_idx]]
        
        # 设置全局先验，使用更保守的范围
        self.global_prior_alpha = torch.clamp(weighted_alpha, min=1.1, max=2.5)
        
        # 记录聚合后的统计信息
        self.logger.info(f"Byzantine-robust aggregation: Min={self.global_prior_alpha.min().item():.2f}, "
                        f"Max={self.global_prior_alpha.max().item():.2f}, "
                        f"Mean={self.global_prior_alpha.mean().item():.2f}")
