import torch
import numpy as np
import time
import os
from copy import deepcopy
from servers.server_base import Server
from clients.client_fedug_v3 import ClientFedUgV3
from utils.util import AverageMeter
from utils.uncertainty_metrics import (
    calculate_brier_score, calculate_ece, plot_reliability_diagram, 
    calculate_ood_auc, calculate_hybrid_ood_score, plot_pr_curve, plot_uncertainty_distribution
)
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

class ServerFedUgV3(Server):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        # 创建客户端
        self.clients = [ClientFedUgV3(args, i) for i in range(args.num_clients)]
        self.client_num = len(self.clients)
        
        # 初始化全局先验 - 降低初始先验强度
        self.global_prior_alpha = torch.ones(self.model.num_classes) * 1.5
        
        # 创建结果目录
        self.result_dir = os.path.join("results", f"fedug_v3_{args.dataset}")
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 记录最佳性能
        self.best_personalized_acc = 0.0
        
        self.logger.info(f"Initialized FedUG V3 Server with {self.client_num} clients")
        self.logger.info(f"Training FedUgV3 with model {args.model_name} on {args.dataset} for {self.client_num} clients.")
        self.logger.info(f"Hyperparameters: LR={args.lr}, WD={args.wd}, Local Epochs={args.local_epochs}, Global Rounds={args.global_rounds}")
    
    def setup_clients(self):
        """设置客户端"""
        clients = []
        for client_idx in range(self.args.n_clients):
            client = ClientFedUgV3(self.args, client_idx)
            clients.append(client)
        return clients
    
    def train(self):
        """训练联邦学习系统"""
        for round_idx in range(1, self.args.global_rounds + 1):
            self.logger.info(f"--- Global Round {round_idx}/{self.args.global_rounds} ---")
            client_indexes = self.client_sampling(round_idx, self.client_num)
            active_clients = [self.clients[i] for i in client_indexes]        
            # 分发全局先验
            self.distribute_prior(client_indexes)

            client_updates = []
            client_alpha_reports = []
            client_data_sizes = []
            
            kl_div_times = AverageMeter()

            for client in active_clients:
                client.set_current_global_round(round_idx)
                stats = client.train_epoch(round_idx)
                
                kl_div_times.update(stats.get('kl_div_time', 0)) # ### SUGGESTION 3: Overhead analysis

                model_weights, alpha_report, data_size = client.get_update()
                
                client_updates.append(model_weights)
                client_alpha_reports.append(alpha_report)
                client_data_sizes.append(data_size)

            avg_kl_time = kl_div_times.avg * 1000 # in ms
            self.logger.info(f"[Overhead Analysis] Avg. KL-Div computation time per batch: {avg_kl_time:.4f} ms")
            # Report communication overhead
            if client_alpha_reports:
                report_size_bytes = client_alpha_reports[0].element_size() * client_alpha_reports[0].nelement()
                self.logger.info(f"[Overhead Analysis] Communication overhead per client for alpha_report: {report_size_bytes} bytes")
            
            self.aggregate_priors(client_indexes)
            # 评估模型
            if round_idx % self.args.eval_gap == 0 or round_idx == self.args.global_rounds:
                self.logger.info(f"--- Evaluating models at round {round_idx} with Advanced Uncertainty Metrics ---")
                self.evaluate(round_idx)
    
    def distribute_prior(self, client_indexes):
        """分发全局先验"""
        # Ensure prior is on CPU before distribution
        cpu_prior = self.global_prior_alpha.cpu()
        for client_idx in client_indexes:
            client = self.clients[client_idx]
            client.global_prior_alpha = cpu_prior.clone()
    
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
        
        # 更新客户端的全局模型
        for client in self.clients:
            client.model_global.load_state_dict(aggregated_state_dict)
    
    def aggregate_priors(self, client_indexes):
        """聚合先验"""
        total_data_size = 0
        alpha_reports = []
        data_sizes = []
        
        for client_idx in client_indexes:
            client = self.clients[client_idx]
            _, alpha_report, data_size = client.get_update()
            
            alpha_reports.append(alpha_report)
            data_sizes.append(data_size)
            total_data_size += data_size
        
        # 加权聚合先验
        new_global_prior = torch.zeros_like(self.global_prior_alpha)
        for i, alpha_report in enumerate(alpha_reports):
            weight = data_sizes[i] / total_data_size
            new_global_prior += alpha_report * weight
        
        # 平滑更新全局先验
        momentum = 0.8  # 动量因子，防止先验剧烈变化
        self.global_prior_alpha = momentum * self.global_prior_alpha + (1 - momentum) * new_global_prior
        
        # 限制先验范围，防止过度自信
        self.global_prior_alpha = torch.clamp(self.global_prior_alpha, min=1.1, max=5.0)
    
    def evaluate(self, round_idx):
        """评估模型性能和不确定性质量"""
        # 收集所有客户端的评估结果
        personal_test_results = []
        global_test_results = []
        ood_results = []
        
        for client_idx in range(self.client_num):
            client = self.clients[client_idx]
            
            # 个性化模型在测试集上的结果
            personal_result = client.get_eval_output(use_personal=True, dataset='test')
            if personal_result:
                personal_test_results.append(personal_result)
            
            # 全局模型在测试集上的结果
            global_result = client.get_eval_output(use_personal=False, dataset='test')
            if global_result:
                global_test_results.append(global_result)
            
            # 个性化模型在OOD数据集上的结果
            ood_result = client.get_eval_output(use_personal=True, dataset='ood')
            if ood_result:
                ood_results.append(ood_result)
        
        # 评估个性化模型性能
        if personal_test_results:
            self.evaluate_model_performance(personal_test_results, "Personalized", round_idx)
        
        # 评估全局模型性能
        if global_test_results:
            self.evaluate_model_performance(global_test_results, "Global", round_idx)
        
        # 评估OOD检测性能
        if personal_test_results and ood_results:
            self.evaluate_ood_detection(personal_test_results, ood_results, round_idx)
    
    def evaluate_model_performance(self, results, model_type, round_idx):
        """评估模型性能和不确定性校准"""
        all_probs = np.concatenate([r['probs'] for r in results])
        all_labels = np.concatenate([r['labels'] for r in results])
        all_epistemic = np.concatenate([r['epistemic_uncertainties'] for r in results])
        all_aleatoric = np.concatenate([r['aleatoric_uncertainties'] for r in results])
        
        # 计算准确率
        preds = np.argmax(all_probs, axis=1)
        accuracy = np.mean(preds == all_labels) * 100
        
        # 计算不确定性校准指标
        brier_score = calculate_brier_score(all_probs, all_labels)
        ece, diagram_data = calculate_ece(all_probs, all_labels, n_bins=15)
        
        # 记录结果
        self.logger.info(f"[Round {round_idx}] {model_type} Model Performance:")
        self.logger.info(f"  - Accuracy: {accuracy:.2f}%")
        self.logger.info(f"  - Brier Score: {brier_score:.4f}")
        self.logger.info(f"  - ECE: {ece:.4f}")
        
        # 分析不确定性与错误的关系
        correct_mask = (preds == all_labels)
        epistemic_correct = all_epistemic[correct_mask]
        epistemic_wrong = all_epistemic[~correct_mask]
        
        if len(epistemic_correct) > 0 and len(epistemic_wrong) > 0:
            # 计算不确定性与错误的相关性
            mean_epistemic_correct = np.mean(epistemic_correct)
            mean_epistemic_wrong = np.mean(epistemic_wrong)
            
            self.logger.info(f"  - Mean Epistemic Uncertainty (Correct): {mean_epistemic_correct:.4f}")
            self.logger.info(f"  - Mean Epistemic Uncertainty (Wrong): {mean_epistemic_wrong:.4f}")
            self.logger.info(f"  - Uncertainty Error Correlation: {mean_epistemic_wrong/mean_epistemic_correct:.2f}x")
            
            # 绘制可靠性图
            reliability_path = os.path.join(self.result_dir, f"{model_type.lower()}_reliability_r{round_idx}.png")
            plot_reliability_diagram(diagram_data, filename=reliability_path)
            self.logger.info(f"  - Reliability diagram saved to {reliability_path}")
            
            # 添加：绘制不确定性分布图
            uncertainty_dist_path = os.path.join(self.result_dir, f"{model_type.lower()}_uncertainty_dist_r{round_idx}.png")
            plot_uncertainty_distribution(
                all_epistemic, all_aleatoric,
                save_path=uncertainty_dist_path
            )
            self.logger.info(f"  - Uncertainty distribution saved to {uncertainty_dist_path}")
        
        # 更新最佳性能
        if model_type == "Personalized" and accuracy > self.best_personalized_acc:
            self.best_personalized_acc = accuracy
            self.logger.info(f"  - New best personalized accuracy: {self.best_personalized_acc:.2f}%")
    
    def evaluate_ood_detection(self, id_results, ood_results, round_idx):
        """评估OOD检测性能"""
        # 收集ID和OOD样本的不确定性和能量分数
        id_epistemic = np.concatenate([r['epistemic_uncertainties'] for r in id_results])
        id_aleatoric = np.concatenate([r['aleatoric_uncertainties'] for r in id_results])
        id_energies = np.concatenate([r['energies'] for r in id_results])
        id_features = np.concatenate([r['features'] for r in id_results])
        
        ood_epistemic = np.concatenate([r['epistemic_uncertainties'] for r in ood_results])
        ood_aleatoric = np.concatenate([r['aleatoric_uncertainties'] for r in ood_results])
        ood_energies = np.concatenate([r['energies'] for r in ood_results])
        ood_features = np.concatenate([r['features'] for r in ood_results])
        
        # 使用认知不确定性评估OOD检测
        epistemic_auroc, epistemic_aupr, epistemic_fpr = calculate_ood_auc(
            id_epistemic, ood_epistemic, pos_label=1
        )
        
        # 使用能量分数评估OOD检测
        energy_auroc, energy_aupr, energy_fpr = calculate_ood_auc(
            id_energies, ood_energies, pos_label=-1
        )
        
        # 使用混合分数评估OOD检测
        id_hybrid = calculate_hybrid_ood_score(id_epistemic, id_energies, id_features)
        ood_hybrid = calculate_hybrid_ood_score(ood_epistemic, ood_energies, ood_features)
        hybrid_auroc, hybrid_aupr, hybrid_fpr = calculate_ood_auc(
            id_hybrid, ood_hybrid, pos_label=1
        )
        
        # 记录结果
        self.logger.info(f"[Round {round_idx}] OOD Detection Performance:")
        self.logger.info(f"  - Epistemic Uncertainty: AUROC={epistemic_auroc:.4f}, AUPR={epistemic_aupr:.4f}, FPR@95%TPR={epistemic_fpr:.4f}")
        self.logger.info(f"  - Energy Score: AUROC={energy_auroc:.4f}, AUPR={energy_aupr:.4f}, FPR@95%TPR={energy_fpr:.4f}")
        self.logger.info(f"  - Hybrid Score: AUROC={hybrid_auroc:.4f}, AUPR={hybrid_aupr:.4f}, FPR@95%TPR={hybrid_fpr:.4f}")
        
        # 绘制PR曲线
        pr_curve_path = os.path.join(self.result_dir, f"ood_pr_curve_r{round_idx}.png")
        plot_pr_curve(
            [id_epistemic, id_energies, id_hybrid],
            [ood_epistemic, ood_energies, ood_hybrid],
            ["Epistemic", "Energy", "Hybrid"],
            save_path=pr_curve_path
        )
        self.logger.info(f"  - PR curve saved to {pr_curve_path}")
        
        # 添加：绘制ID和OOD不确定性分布对比图
        uncertainty_dist_path = os.path.join(self.result_dir, f"ood_uncertainty_dist_r{round_idx}.png")
        plot_uncertainty_distribution(
            id_epistemic, id_aleatoric,
            ood_epistemic, ood_aleatoric,
            save_path=uncertainty_dist_path
        )
        self.logger.info(f"  - ID/OOD uncertainty distribution saved to {uncertainty_dist_path}")
        
        # 分析不确定性分布
        self.logger.info(f"  - ID Epistemic Uncertainty: Mean={np.mean(id_epistemic):.4f}, Std={np.std(id_epistemic):.4f}")
        self.logger.info(f"  - OOD Epistemic Uncertainty: Mean={np.mean(ood_epistemic):.4f}, Std={np.std(ood_epistemic):.4f}")
        self.logger.info(f"  - Uncertainty Separation: {(np.mean(ood_epistemic) - np.mean(id_epistemic)) / np.std(id_epistemic):.2f} sigma")
    
    def client_sampling(self, round_idx, client_num):
        """客户端采样策略"""
        num_clients = max(int(self.sampling_prob * client_num), 1)
        
        # 基于不确定性的采样策略
        if hasattr(self.args, 'uncertainty_sampling') and self.args.uncertainty_sampling and round_idx > 1:
            # 收集所有客户端的平均不确定性
            client_uncertainties = []
            for client_idx in range(client_num):
                client = self.clients[client_idx]
                result = client.get_eval_output(use_personal=True, dataset='test')
                if result:
                    mean_uncertainty = np.mean(result['epistemic_uncertainties'])
                    client_uncertainties.append((client_idx, mean_uncertainty))
                else:
                    client_uncertainties.append((client_idx, 0.0))
            
            # 按不确定性排序
            client_uncertainties.sort(key=lambda x: x[1], reverse=True)
            
            # 选择不确定性最高的客户端
            high_uncertainty_ratio = 0.7  # 高不确定性客户端的比例
            high_uncertainty_count = int(num_clients * high_uncertainty_ratio)
            
            # 高不确定性客户端
            high_uncertainty_clients = [x[0] for x in client_uncertainties[:high_uncertainty_count]]
            
            # 随机选择剩余客户端
            remaining_clients = [x[0] for x in client_uncertainties[high_uncertainty_count:]]
            np.random.shuffle(remaining_clients)
            random_clients = remaining_clients[:num_clients - high_uncertainty_count]
            
            # 合并选择的客户端
            selected_clients = high_uncertainty_clients + random_clients
        else:
            # 随机采样
            selected_clients = np.random.choice(range(client_num), num_clients, replace=False)
        
        return selected_clients
