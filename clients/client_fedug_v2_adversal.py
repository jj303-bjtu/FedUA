import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
import time 
import os
import matplotlib.pyplot as plt
from clients.client_base import Client
from utils.util import AverageMeter
from data.utils.loader import get_partition

class ClientFedUgV2(Client):
    """
    <<< FINAL VERSION >>>
    优化版FedUG客户端:
    1. 通过标签平滑改善模型校准 (降低ECE)
    2. 通过对抗训练提升OOD检测能力，且不依赖OOD训练数据
    3. 多种不确定性估计方法集成，提高鲁棒性
    4. 特征空间正则化，改善表示学习
    5. 自适应温度校准，提高预测可靠性
    """
    def __init__(self, args, client_idx, is_corrupted=False):
        super().__init__(args, client_idx, is_corrupted)
        self.args = args
        
        self.model_global = deepcopy(self.model)
        self.model_personal = deepcopy(self.model) 
        
        # 不确定性估计参数优化
        self.uncertainty_threshold = 0.8  # 更合理的阈值
        self.max_evidence = 2.0  # 限制evidence上限，防止过度自信
        self.edl_weight = 0.3  # 降低EDL权重，避免主导训练
        
        # 温度参数优化
        self.temperature = 1.2  # 基础温度值
        self.use_temperature_ensemble = True
        self.temperature_values = [0.8, 1.2, 2.0]  # 多温度集成
        self.adaptive_temp = True  # 自适应温度校准
        
        # 不确定性估计方法
        self.use_energy_uncertainty = True
        self.energy_weight = 0.4
        self.use_mutual_info = True  # 使用互信息作为不确定性度量
        self.mi_weight = 0.3
        
        # 对抗训练参数优化
        self.use_adversarial_ood = args.use_adv_ood if hasattr(args, 'use_adv_ood') else True
        self.adv_epsilon = 0.03  # 扰动强度
        self.adv_steps = 2  # PGD步数
        self.adv_alpha = 0.01  # PGD步长
        self.adv_ood_reg_weight = 0.8  # OOD正则化权重
        
        # 特征空间正则化
        self.use_feature_regularization = True
        self.feature_weight = 0.3
        
        # 全局一致性和先验
        self.global_prior_alpha = None
        self.current_round = 0
        self.global_consistency_weight = args.global_consistency_weight if hasattr(args, 'global_consistency_weight') else 0.1
        
        # 学习率设置
        self.base_lr = args.lr
        self.personal_lr_factor = 0.2
        
        # 标签平滑
        self.label_smoothing = args.label_smoothing if hasattr(args, 'label_smoothing') else 0.1
        self.ce_loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # 类特征中心
        self.class_centroids = None
        
        # 不确定性指标记录
        self.uncertainty_metrics = {
            'epistemic': [],
            'aleatoric': [],
            'energy': [],
            'mutual_info': [],
            'ece': [],
            'auroc': []
        }
        
        # 创建结果保存目录
        self.results_dir = f"results/client_{client_idx}"
        os.makedirs(self.results_dir, exist_ok=True)

    def set_global_prior_alpha(self, global_prior_alpha):
        if global_prior_alpha is not None:
            self.global_prior_alpha = global_prior_alpha.clone().detach().to(self.device)
    
    def set_current_global_round(self, current_round):
        self.current_round = current_round
    
    def _ensemble_uncertainty(self, logits):
        """使用多温度集成提高不确定性估计的鲁棒性"""
        if not self.use_temperature_ensemble:
            evidence = F.softplus(logits / self.temperature)
            evidence = torch.clamp(evidence, min=1e-6, max=self.max_evidence)
            alpha = evidence + 1.0
            return alpha
        
        # 多温度集成
        alphas = []
        weights = []
        
        for i, temp in enumerate(self.temperature_values):
            evidence = F.softplus(logits / temp)
            evidence = torch.clamp(evidence, min=1e-6, max=self.max_evidence)
            alpha = evidence + 1.0
            alphas.append(alpha)
            
            # 自适应权重 - 基于预测分布的熵
            if self.adaptive_temp:
                probs = F.softmax(logits / temp, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                weights.append(torch.exp(-entropy))
            else:
                weights.append(torch.tensor(1.0 / len(self.temperature_values)))
        
        # 归一化权重
        if self.adaptive_temp:
            weights = [w / sum(weights) for w in weights]
        
        # 加权平均
        ensemble_alpha = torch.zeros_like(alphas[0])
        for i, alpha in enumerate(alphas):
            ensemble_alpha += weights[i] * alpha
        
        return ensemble_alpha

    def calculate_evidence_loss(self, logits, targets, features=None, is_personal=False):
        """增强版证据理论损失函数，集成多种不确定性估计方法"""
        num_classes = logits.size(1)
        
        # 使用多温度集成获取alpha
        alpha = self._ensemble_uncertainty(logits)
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # 温度校准的交叉熵损失
        calibrated_logits = logits / self.temperature
        ce_loss = self.ce_loss_fn(calibrated_logits, targets)
        
        # 平滑标签
        y_one_hot = F.one_hot(targets, num_classes).float()
        if self.label_smoothing > 0:
            y_one_hot = y_one_hot * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        # EDL对数似然损失
        log_likelihood = torch.sum(y_one_hot * (torch.digamma(alpha) - torch.digamma(S)), dim=1)
        nll_loss = -log_likelihood.mean()
        
        # 类别不确定性损失
        present_classes = (y_one_hot.sum(dim=0) > 0)
        seen_alpha = alpha[:, present_classes]
        uncertainty_loss_seen = (num_classes / torch.sum(seen_alpha, dim=1)).mean()
        
        # 未见类别不确定性损失
        uncertainty_loss_unseen = 0.0
        if present_classes.sum() < num_classes:
            unseen_mask = ~present_classes
            unseen_alpha = alpha[:, unseen_mask]
            unseen_uncertainty = torch.abs((unseen_mask.sum().float() / torch.sum(unseen_alpha, dim=1)) - self.uncertainty_threshold)
            uncertainty_loss_unseen = unseen_uncertainty.mean()
        
        # 全局一致性KL散度
        kl_div_val = 0.0
        kl_computation_time = 0.0
        if is_personal and self.global_prior_alpha is not None:
            prior_alpha = self.global_prior_alpha.unsqueeze(0).expand_as(alpha)
            prior_alpha = torch.clamp(prior_alpha, min=1.0, max=self.max_evidence + 1.0)
            
            start_time = time.time()
            kl_div = self._safe_kl_divergence(alpha, prior_alpha)
            kl_computation_time = time.time() - start_time
            kl_div_val = kl_div
        
        # 能量不确定性损失
        energy_loss = 0.0
        if self.use_energy_uncertainty:
            energy = -torch.logsumexp(logits / self.temperature, dim=1)
            
            # 分离正确和错误预测
            correct_mask = torch.argmax(logits, dim=1) == targets
            
            if correct_mask.any():
                energy_correct = energy[correct_mask]
                energy_loss += torch.mean(energy_correct)
            
            if (~correct_mask).any():
                energy_incorrect = energy[~correct_mask]
                energy_loss -= torch.mean(energy_incorrect)
        
        # 互信息不确定性
        mi_loss = 0.0
        if self.use_mutual_info:
            probs = F.softmax(logits / self.temperature, dim=1)
            expected_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            
            # 计算预测分布的熵
            entropy_expected = -torch.sum((alpha / S) * (torch.digamma(alpha + 1) - torch.digamma(S + 1)), dim=1).mean()
            
            # 互信息 = 总不确定性 - 偶然不确定性
            mutual_info = expected_entropy - entropy_expected
            mi_loss = -mutual_info  # 最大化互信息
        
        # 特征空间正则化
        feature_loss = 0.0
        if self.use_feature_regularization and features is not None and self.class_centroids is not None:
            # 计算特征与类中心的距离
            for i in range(features.size(0)):
                label = targets[i].item()
                if label < len(self.class_centroids):
                    centroid = self.class_centroids[label].to(self.device)
                    feature = features[i]
                    # 鼓励特征接近其类中心
                    feature_loss += F.mse_loss(feature, centroid)
        
        # 组合损失
        total_loss = (
            ce_loss + 
            self.edl_weight * nll_loss +
            0.2 * uncertainty_loss_seen +  
            0.1 * uncertainty_loss_unseen +  
            self.global_consistency_weight * kl_div_val +
            self.energy_weight * energy_loss +
            self.mi_weight * mi_loss +
            self.feature_weight * feature_loss
        )
        
        return total_loss, ce_loss, nll_loss, alpha, S, kl_computation_time

    def _safe_kl_divergence(self, alpha1, alpha2):
        """安全的KL散度计算，防止数值不稳定"""
        sum_alpha1 = torch.sum(alpha1, dim=1)
        sum_alpha2 = torch.sum(alpha2, dim=1)
        
        alpha1_safe = torch.clamp(alpha1, min=1e-6, max=100.0)
        alpha2_safe = torch.clamp(alpha2, min=1e-6, max=100.0)
        sum_alpha1_safe = torch.clamp(sum_alpha1, min=1e-6, max=1000.0)
        sum_alpha2_safe = torch.clamp(sum_alpha2, min=1e-6, max=1000.0)
        
        term1 = torch.lgamma(sum_alpha1_safe) - torch.lgamma(sum_alpha2_safe)
        term2 = torch.sum(torch.lgamma(alpha2_safe) - torch.lgamma(alpha1_safe), dim=1)
        term3 = torch.sum((alpha1_safe - alpha2_safe) * (torch.digamma(alpha1_safe) - torch.digamma(sum_alpha1_safe.unsqueeze(1))), dim=1)
        
        kl = term1 + term2 + term3
        return torch.clamp(kl, min=0.0).mean()
    
    def train(self, round_idx):
        """训练客户端模型"""
        self.set_current_global_round(round_idx)
        self._train_global_model()
        stats = self._train_personal_model(round_idx)
        
        # 每10轮保存一次不确定性指标图
        if round_idx % 10 == 0:
            self._save_uncertainty_metrics(round_idx)
        
        return stats
    
    def _extract_features(self, x):
        """从模型中提取特征"""
        # 获取倒数第二层特征
        if hasattr(self.model_personal, 'features'):
            return self.model_personal.features(x)
        elif hasattr(self.model_personal, 'backbone'):
            return self.model_personal.backbone(x)
        elif hasattr(self.model_personal, 'feature_extractor'):
            return self.model_personal.feature_extractor(x)
        else:
            # 如果没有明确的特征提取器，尝试获取最后一层之前的输出
            # 这需要根据具体模型结构调整
            return None
    
    def _train_global_model(self):
        """训练全局模型"""
        self.model_global.train()
        self.model_global.to(self.device)
        optimizer = torch.optim.SGD(
            self.model_global.parameters(), 
            lr=self.base_lr, 
            momentum=0.9, 
            weight_decay=self.args.wd
        )
        
        for epoch in range(self.args.local_epochs):
            train_loader = self.load_train_data()
            if not train_loader: continue
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                # 提取特征（如果可能）
                features = self._extract_features(x)
                
                # 前向传播
                logits = self.model_global(x)
                
                # 计算损失
                edl_loss, ce_loss, _, _, _, _ = self.calculate_evidence_loss(
                    logits, y, features, is_personal=False
                )
                
                # 组合损失
                total_loss = 0.7 * ce_loss + 0.3 * edl_loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_global.parameters(), max_norm=1.0)
                optimizer.step()
        
        self.model_global.to("cpu")
    
    def _collect_class_centroids(self, train_loader):
        """收集每个类的特征中心"""
        self.model_personal.eval()
        self.model_personal.to(self.device)
        
        class_features = {}
        
        with torch.no_grad():
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # 提取特征
                features = self._extract_features(x)
                if features is None:
                    continue
                    
                features = features.view(features.size(0), -1)  # 展平特征
                
                # 按类别收集特征
                for i in range(x.size(0)):
                    label = y[i].item()
                    if label not in class_features:
                        class_features[label] = []
                    class_features[label].append(features[i].cpu())
        
        # 计算每个类的中心
        centroids = []
        for c in range(self.num_classes):
            if c in class_features and len(class_features[c]) > 0:
                centroids.append(torch.stack(class_features[c]).mean(0))
            else:
                # 如果没有这个类的样本，使用零向量
                if len(centroids) > 0:
                    centroids.append(torch.zeros_like(centroids[0]))
                else:
                    # 如果是第一个类，无法确定向量维度，跳过
                    pass
        
        if len(centroids) > 0:
            self.class_centroids = torch.stack(centroids)
        
        self.model_personal.train()
    
    def _generate_adversarial_samples(self, x, y):
        """生成对抗样本，使用PGD方法"""
        self.model_personal.eval()
        
        # 初始化对抗样本
        x_adv = x.clone().detach()
        
        # 添加随机初始化
        x_adv = x_adv + torch.zeros_like(x_adv).uniform_(-self.adv_epsilon/2, self.adv_epsilon/2)
        x_adv = torch.clamp(x_adv, 0, 1)
        
        # 多步PGD
        for _ in range(self.adv_steps):
            x_adv.requires_grad = True
            
            # 前向传播
            logits = self.model_personal(x_adv)
            
            # 计算损失 - 最小化正确类别的概率
            loss = -F.cross_entropy(logits, y)
            
            # 反向传播
            self.model_personal.zero_grad()
            loss.backward()
            
            # PGD更新
            with torch.no_grad():
                grad = x_adv.grad.sign()
                x_adv = x_adv + self.adv_alpha * grad
                # 投影回epsilon球
                x_adv = torch.min(torch.max(x_adv, x - self.adv_epsilon), x + self.adv_epsilon)
                x_adv = torch.clamp(x_adv, 0, 1)
        
        self.model_personal.train()
        return x_adv.detach()
    
    def _train_personal_model(self, round_idx):
        """训练个性化模型"""
        # 从全局模型初始化个性化模型
        self.model_personal.load_state_dict(self.model_global.state_dict())
        self.model_personal.train()
        self.model_personal.to(self.device)
        
        # 保存全局参数用于正则化
        global_params = {name: param.clone().detach() for name, param in self.model_global.named_parameters()}
        
        # 优化器
        optimizer = torch.optim.SGD(
            self.model_personal.parameters(), 
            lr=self.base_lr * self.personal_lr_factor, 
            momentum=0.9, 
            weight_decay=self.args.wd * 1.5
        )
        
        # 指标记录
        acc_meter = AverageMeter()
        kl_time_meter = AverageMeter()
        uncertainty_meters = {k: AverageMeter() for k in self.uncertainty_metrics.keys()}
        
        # 加载训练数据
        train_loader = self.load_train_data()
        if not train_loader: 
            return {'acc': 0, 'kl_div_time': 0}
        
        # 收集类特征中心
        self._collect_class_centroids(train_loader)
        
        # 个性化训练轮次
        personal_epochs = max(3, self.args.local_epochs // 2)
        
        for epoch in range(personal_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # 提取特征
                features = self._extract_features(x)
                
                # --- 计算ID样本损失 ---
                logits = self.model_personal(x)
                edl_loss, ce_loss, nll_loss, alpha, S, kl_time = self.calculate_evidence_loss(
                    logits, y, features, is_personal=True
                )
                kl_time_meter.update(kl_time)
                
                # --- 近端正则化 ---
                prox_loss = 0.0
                for name, param in self.model_personal.named_parameters():
                    if name in global_params:
                        prox_loss += ((param - global_params[name].to(self.device)) ** 2).sum()
                
                # --- 对抗OOD正则化 ---
                adv_ood_loss = torch.tensor(0.0).to(self.device)
                if self.use_adversarial_ood and self.adv_ood_reg_weight > 0:
                    # 生成对抗样本
                    x_adv = self._generate_adversarial_samples(x, y)
                    
                    # 计算对抗样本的不确定性损失
                    logits_adv = self.model_personal(x_adv)
                    
                    # 计算多种不确定性指标
                    evidence_adv = F.softplus(logits_adv / self.temperature)
                    alpha_adv = evidence_adv + 1.0
                    S_adv = torch.sum(alpha_adv, dim=1)
                    
                    # 认知不确定性 (epistemic)
                    epistemic_adv = self.num_classes / S_adv
                    
                    # 偶然不确定性 (aleatoric) - 使用熵
                    probs_adv = F.softmax(logits_adv / self.temperature, dim=1)
                    aleatoric_adv = -torch.sum(probs_adv * torch.log(probs_adv + 1e-8), dim=1)
                    
                    # 能量不确定性
                    energy_adv = -torch.logsumexp(logits_adv / self.temperature, dim=1)
                    energy_uncertainty = torch.sigmoid(energy_adv)
                    
                    # 互信息不确定性
                    expected_entropy = -torch.sum(probs_adv * torch.log(probs_adv + 1e-8), dim=1)
                    entropy_expected = -torch.sum((alpha_adv / S_adv.unsqueeze(1)) * 
                                                (torch.digamma(alpha_adv + 1) - torch.digamma(S_adv.unsqueeze(1) + 1)), dim=1)
                    mutual_info = expected_entropy - entropy_expected
                    
                    # 组合不确定性损失 - 对抗样本应该有高不确定性
                    adv_ood_loss = -(
                        1.2 * epistemic_adv.mean() +  # 认知不确定性
                        0.8 * aleatoric_adv.mean() +  # 偶然不确定性
                        0.8 * energy_uncertainty.mean() +  # 能量不确定性
                        1.0 * mutual_info.mean()  # 互信息
                    )
                    
                    # 更新不确定性指标
                    uncertainty_meters['epistemic'].update(epistemic_adv.mean().item())
                    uncertainty_meters['aleatoric'].update(aleatoric_adv.mean().item())
                    uncertainty_meters['energy'].update(energy_uncertainty.mean().item())
                    uncertainty_meters['mutual_info'].update(mutual_info.mean().item())
                
                # --- 组合所有损失 ---
                total_loss = (
                    edl_loss + 
                    0.01 * prox_loss + 
                    self.adv_ood_reg_weight * adv_ood_loss
                )
                
                # 反向传播和优化
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_personal.parameters(), max_norm=0.5)
                optimizer.step()
                
                # 更新指标
                with torch.no_grad():
                    pred = F.softmax(logits / self.temperature, dim=1).argmax(1)
                    acc = (pred == y).float().mean() * 100
                    acc_meter.update(acc.item(), x.size(0))
        
        # 保存不确定性指标
        for k, meter in uncertainty_meters.items():
            if meter.count > 0:
                self.uncertainty_metrics[k].append(meter.avg)
        
        self.model_personal.to("cpu")
        
        return {
            'acc': acc_meter.avg,
            'kl_div_time': kl_time_meter.avg if kl_time_meter.count > 0 else 0.0
        }
    
    def _save_uncertainty_metrics(self, round_idx):
        """保存不确定性指标图"""
        if not any(len(v) > 0 for v in self.uncertainty_metrics.values()):
            return
            
        plt.figure(figsize=(12, 8))
        
        for name, values in self.uncertainty_metrics.items():
            if len(values) > 0:
                plt.plot(range(len(values)), values, label=name)
        
        plt.title(f'Uncertainty Metrics Evolution (Client {self.client_idx})')
        plt.xlabel('Training Rounds')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True)
        
        # 保存图像
        save_path = os.path.join(self.results_dir, f'uncertainty_metrics_round_{round_idx}.png')
        plt.savefig(save_path)
        plt.close()
    
    def get_update(self):
        """获取客户端更新"""
        model_weights = deepcopy(self.model_global.state_dict())
        alpha_report = self._compute_alpha_report()
        data_size = self.num_train
        return model_weights, alpha_report, data_size
    
    def _compute_alpha_report(self):
        """计算并报告alpha值"""
        self.model_global.eval()
        self.model_global.to(self.device)
        
        train_loader = self.load_train_data()
        if not train_loader: 
            return torch.ones(self.num_classes) * 1.5
        
        # 使用raw_evidence策略
        if not hasattr(self.args, 'prior_report_strategy') or self.args.prior_report_strategy == 'raw_evidence':
            all_evidence = []
            with torch.no_grad():
                for x, _ in train_loader:
                    x = x.to(self.device)
                    logits = self.model_global(x)
                    
                    # 使用多温度集成
                    if self.use_temperature_ensemble:
                        alpha = self._ensemble_uncertainty(logits)
                        evidence = alpha - 1.0
                    else:
                        evidence = F.softplus(logits / self.temperature)
                        evidence = torch.clamp(evidence, min=1e-6, max=self.max_evidence)
                    
                    all_evidence.append(evidence.cpu())
            
            if not all_evidence: 
                return torch.ones(self.num_classes) * 1.5
                
            avg_evidence = torch.cat(all_evidence, dim=0).mean(dim=0)
            # 使用更保守的evidence估计
            alpha_report = avg_evidence * 0.7 + 1.0  # 缩小evidence影响
        
        # 使用softmax_prob策略
        else:  
            all_probs = []
            with torch.no_grad():
                for x, _ in train_loader:
                    x = x.to(self.device)
                    logits = self.model_global(x)
                    probs = F.softmax(logits / self.temperature, dim=1)
                    all_probs.append(probs.cpu())
            
            if not all_probs: 
                return torch.ones(self.num_classes) * 1.5
                
            avg_probs = torch.cat(all_probs, dim=0).mean(dim=0)
            # 使用更保守的先验强度
            prior_strength = min(2.0, getattr(self.args, 'prior_strength', 1.0))
            alpha_report = avg_probs * prior_strength + 1.0
        
        self.model_global.to("cpu")
        # 更严格的上限，防止过度自信
        return torch.clamp(alpha_report, min=1.1, max=2.0)

    def get_eval_output(self, use_personal=True, dataset='test'):
        """获取评估输出，包括不确定性指标"""
        model_eval = self.model_personal if use_personal else self.model_global
        model_eval.eval()
        model_eval.to(self.device)
        
        if dataset == 'test':
            loader = self.load_test_data()
        elif dataset == 'ood':
            loader = self.load_ood_data()
        else:
            return None

        if not loader:
            return None

        all_probs = []
        all_labels = []
        all_epistemic_uncertainties = []
        all_aleatoric_uncertainties = []
        all_energy_uncertainties = []
        all_mutual_info = []
        all_total_uncertainties = []
        all_features = []
        
        with torch.no_grad():
            if len(loader) == 0:
                return None
                
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # 提取特征
                features = self._extract_features(x)
                if features is not None:
                    features = features.view(features.size(0), -1)
                    all_features.append(features.cpu().numpy())
                
                logits = model_eval(x)
                
                # 使用多温度集成
                if self.use_temperature_ensemble:
                    alpha = self._ensemble_uncertainty(logits)
                    evidence = alpha - 1.0
                else:
                    evidence = F.softplus(logits / self.temperature)
                    evidence = torch.clamp(evidence, min=1e-6, max=self.max_evidence)
                    alpha = evidence + 1.0
                
                S = torch.sum(alpha, dim=1, keepdim=True)
                
                # 计算概率
                probs = F.softmax(logits / self.temperature, dim=1)
                
                # 计算认知不确定性 (epistemic)
                epistemic_uncertainty = self.num_classes / torch.sum(evidence, dim=1)
                
                # 计算偶然不确定性 (aleatoric) - 使用熵
                aleatoric_uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                
                # 计算能量不确定性
                energy = -torch.logsumexp(logits / self.temperature, dim=1)
                energy_uncertainty = torch.sigmoid(energy)
                
                # 计算互信息不确定性
                expected_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                entropy_expected = -torch.sum((alpha / S) * (torch.digamma(alpha + 1) - torch.digamma(S + 1)), dim=1)
                mutual_info = expected_entropy - entropy_expected
                
                # 计算总不确定性 (组合多种不确定性)
                total_uncertainty = (
                    0.4 * epistemic_uncertainty + 
                    0.3 * aleatoric_uncertainty + 
                    0.2 * energy_uncertainty + 
                    0.1 * mutual_info
                )
                
                # 收集结果
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                all_epistemic_uncertainties.append(epistemic_uncertainty.cpu().numpy())
                all_aleatoric_uncertainties.append(aleatoric_uncertainty.cpu().numpy())
                all_energy_uncertainties.append(energy_uncertainty.cpu().numpy())
                all_mutual_info.append(mutual_info.cpu().numpy())
                all_total_uncertainties.append(total_uncertainty.cpu().numpy())
        
        model_eval.to("cpu")
        
        if not all_labels:
            return None
        
        result = {
            'probs': np.concatenate(all_probs),
            'labels': np.concatenate(all_labels),
            'epistemic_uncertainties': np.concatenate(all_epistemic_uncertainties),
            'aleatoric_uncertainties': np.concatenate(all_aleatoric_uncertainties),
            'energy_uncertainties': np.concatenate(all_energy_uncertainties),
            'mutual_info': np.concatenate(all_mutual_info),
            'total_uncertainties': np.concatenate(all_total_uncertainties)
        }
        
        if all_features:
            result['features'] = np.concatenate(all_features)
            
        # 计算校准误差 (ECE)
        try:
            from sklearn.metrics import brier_score_loss
            from sklearn.calibration import calibration_curve
            
            probs_np = result['probs']
            labels_np = result['labels']
            
            # 计算Brier分数
            y_one_hot = np.zeros((labels_np.size, self.num_classes))
            y_one_hot[np.arange(labels_np.size), labels_np.astype(int)] = 1
            brier_scores = []
            for i in range(self.num_classes):
                brier_scores.append(brier_score_loss(y_one_hot[:, i], probs_np[:, i]))
            result['brier_score'] = np.mean(brier_scores)
            
            # 计算ECE
            pred_class = np.argmax(probs_np, axis=1)
            confidence = np.max(probs_np, axis=1)
            accuracy = (pred_class == labels_np)
            
            # 10个置信度区间
            n_bins = 10
            bin_indices = np.linspace(0, 1, n_bins + 1)
            ece = 0.0
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for i in range(n_bins):
                bin_start = bin_indices[i]
                bin_end = bin_indices[i + 1]
                if i == n_bins - 1:  # 最后一个区间包含上界
                    mask = (confidence >= bin_start) & (confidence <= bin_end)
                else:
                    mask = (confidence >= bin_start) & (confidence < bin_end)
                
                if np.sum(mask) > 0:
                    bin_acc = np.mean(accuracy[mask])
                    bin_conf = np.mean(confidence[mask])
                    bin_count = np.sum(mask)
                    
                    bin_accuracies.append(bin_acc)
                    bin_confidences.append(bin_conf)
                    bin_counts.append(bin_count)
                    
                    ece += (bin_count / len(accuracy)) * np.abs(bin_acc - bin_conf)
            
            result['ece'] = ece
            result['bin_accuracies'] = np.array(bin_accuracies)
            result['bin_confidences'] = np.array(bin_confidences)
            result['bin_counts'] = np.array(bin_counts)
            
            # 如果是测试集，保存校准图
            if dataset == 'test' and use_personal:
                self._save_calibration_plot(result, round_idx=self.current_round)
                self._save_uncertainty_distribution_plot(result, dataset, round_idx=self.current_round)
        except Exception as e:
            print(f"计算校准误差时出错: {e}")
        
        return result
    
    def _save_calibration_plot(self, result, round_idx):
        """保存校准图"""
        if 'bin_accuracies' not in result or 'bin_confidences' not in result:
            return
            
        plt.figure(figsize=(10, 8))
        
        # 绘制校准曲线
        bin_accuracies = result['bin_accuracies']
        bin_confidences = result['bin_confidences']
        
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(bin_confidences, bin_accuracies, 'o-', label=f'Model Calibration (ECE: {result["ece"]:.4f})')
        
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title(f'Calibration Curve (Client {self.client_idx}, Round {round_idx})')
        plt.legend()
        plt.grid(True)
        
        # 保存图像
        save_path = os.path.join(self.results_dir, f'calibration_round_{round_idx}.png')
        plt.savefig(save_path)
        plt.close()
    
    def _save_uncertainty_distribution_plot(self, result, dataset, round_idx):
        """保存不确定性分布图"""
        plt.figure(figsize=(15, 10))
        
        # 创建2x2子图
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 认知不确定性分布
        axs[0, 0].hist(result['epistemic_uncertainties'], bins=30, alpha=0.7)
        axs[0, 0].set_title('Epistemic Uncertainty Distribution')
        axs[0, 0].set_xlabel('Epistemic Uncertainty')
        axs[0, 0].set_ylabel('Frequency')
        axs[0, 0].grid(True)
        
        # 2. 偶然不确定性分布
        axs[0, 1].hist(result['aleatoric_uncertainties'], bins=30, alpha=0.7)
        axs[0, 1].set_title('Aleatoric Uncertainty Distribution')
        axs[0, 1].set_xlabel('Aleatoric Uncertainty')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].grid(True)
        
        # 3. 能量不确定性分布
        axs[1, 0].hist(result['energy_uncertainties'], bins=30, alpha=0.7)
        axs[1, 0].set_title('Energy Score Distribution')
        axs[1, 0].set_xlabel('Energy Score')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 0].grid(True)
        
        # 4. 互信息不确定性分布
        axs[1, 1].hist(result['mutual_info'], bins=30, alpha=0.7)
        axs[1, 1].set_title('Mutual Information Distribution')
        axs[1, 1].set_xlabel('Mutual Information')
        axs[1, 1].set_ylabel('Frequency')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        dataset_name = 'test' if dataset == 'test' else 'ood'
        save_path = os.path.join(self.results_dir, f'uncertainty_dist_{dataset_name}_round_{round_idx}.png')
        plt.savefig(save_path)
        plt.close()
    
    def evaluate_uncertainty_metrics(self, id_data=None, ood_data=None):
        """评估不确定性指标的质量"""
        if id_data is None or ood_data is None:
            id_results = self.get_eval_output(use_personal=True, dataset='test')
            ood_results = self.get_eval_output(use_personal=True, dataset='ood')
            
            if id_results is None or ood_results is None:
                return None
                
            id_uncertainties = {
                'epistemic': id_results['epistemic_uncertainties'],
                'aleatoric': id_results['aleatoric_uncertainties'],
                'energy': id_results['energy_uncertainties'],
                'mutual_info': id_results['mutual_info'],
                'total': id_results['total_uncertainties']
            }
            
            ood_uncertainties = {
                'epistemic': ood_results['epistemic_uncertainties'],
                'aleatoric': ood_results['aleatoric_uncertainties'],
                'energy': ood_results['energy_uncertainties'],
                'mutual_info': ood_results['mutual_info'],
                'total': ood_results['total_uncertainties']
            }
        else:
            id_uncertainties = id_data
            ood_uncertainties = ood_data
        
        # 计算AUROC和AUPR
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            
            results = {}
            
            for metric_name in id_uncertainties.keys():
                # 准备标签 (0=ID, 1=OOD)
                id_values = id_uncertainties[metric_name]
                ood_values = ood_uncertainties[metric_name]
                
                y_true = np.concatenate([np.zeros(len(id_values)), np.ones(len(ood_values))])
                y_score = np.concatenate([id_values, ood_values])
                
                # 计算AUROC
                auroc = roc_auc_score(y_true, y_score)
                
                # 计算AUPR
                aupr = average_precision_score(y_true, y_score)
                
                results[f'{metric_name}_auroc'] = auroc
                results[f'{metric_name}_aupr'] = aupr
            
            # 保存OOD检测性能图
            self._save_ood_detection_plot(id_uncertainties, ood_uncertainties, results, round_idx=self.current_round)
            
            return results
        except Exception as e:
            print(f"评估不确定性指标时出错: {e}")
            return None
    
    def _save_ood_detection_plot(self, id_uncertainties, ood_uncertainties, metrics, round_idx):
        """保存OOD检测性能图"""
        plt.figure(figsize=(15, 10))
        
        # 创建2x2子图
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        uncertainty_types = ['epistemic', 'aleatoric', 'energy', 'total']
        titles = ['Epistemic Uncertainty', 'Aleatoric Uncertainty', 'Energy Score', 'Total Uncertainty']
        
        for i, (u_type, title) in enumerate(zip(uncertainty_types, titles)):
            row, col = i // 2, i % 2
            
            if u_type in id_uncertainties and u_type in ood_uncertainties:
                id_values = id_uncertainties[u_type]
                ood_values = ood_uncertainties[u_type]
                
                # 绘制ID和OOD的不确定性分布
                axs[row, col].hist(id_values, bins=30, alpha=0.5, label='ID', density=True)
                axs[row, col].hist(ood_values, bins=30, alpha=0.5, label='OOD', density=True)
                
                auroc = metrics.get(f'{u_type}_auroc', 0)
                aupr = metrics.get(f'{u_type}_aupr', 0)
                
                axs[row, col].set_title(f'{title} (AUROC: {auroc:.4f}, AUPR: {aupr:.4f})')
                axs[row, col].set_xlabel('Uncertainty Value')
                axs[row, col].set_ylabel('Density')
                axs[row, col].legend()
                axs[row, col].grid(True)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.results_dir, f'ood_detection_round_{round_idx}.png')
        plt.savefig(save_path)
        plt.close()
