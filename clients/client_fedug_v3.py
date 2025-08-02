import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from copy import deepcopy
from clients.client_base import Client
from utils.util import AverageMeter
import numpy as np

class ClientFedUgV3(Client):
    """
    FedUG V3: 深度不确定性重构版
    1. 多头不确定性估计 - 使用集成方法提升鲁棒性
    2. 对比学习增强 - 提高特征区分能力
    3. 能量模型集成 - 提供更可靠的OOD检测
    4. 特征空间正则化 - 改善不确定性校准
    """
    def __init__(self, args, client_idx, is_corrupted=False):
        super().__init__(args, client_idx, is_corrupted)
        self.args = args
        
        # 模型初始化
        self.model_global = deepcopy(self.model)
        self.model_personal = deepcopy(self.model)
        
        # 不确定性估计参数 - 调整参数
        self.temperature = 1.5  # 增加温度参数
        self.max_evidence = 5.0  # 降低最大evidence值
        self.num_classes = 10  # 类别数量
        
        # 多头不确定性估计 - 简化设计
        self.num_heads = 3
        self.head_weights = nn.Parameter(torch.ones(self.num_heads) / self.num_heads)
        
        # 对比学习参数 - 调整权重
        self.use_contrastive = True
        self.contrastive_temp = 0.2  # 增加温度
        self.contrastive_weight = 0.1  # 降低权重
        
        # 能量模型参数
        self.energy_temp = 1.5  # 增加温度
        self.energy_margin = 5.0  # 降低边界值
        
        # 特征空间正则化
        self.feature_reg_weight = 0.05  # 降低权重
        
        # 学习率设置
        self.base_lr = args.lr
        self.personal_lr_factor = 0.1
        
        # 损失函数
        self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 全局先验
        self.global_prior_alpha = None
        self.current_round = 0
    
    def _safe_kl_divergence(self, alpha, prior_alpha):
        """安全的KL散度计算"""
        # Ensure both tensors are on the same device
        device = alpha.device
        prior_alpha = prior_alpha.to(device)
        
        beta = torch.sum(alpha, dim=1, keepdim=True)
        prior_beta = torch.sum(prior_alpha, dim=1, keepdim=True)
        
        term1 = torch.lgamma(beta) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        term2 = torch.sum((alpha - prior_alpha) * (torch.digamma(alpha) - torch.digamma(beta)), dim=1, keepdim=True)
        term3 = torch.lgamma(prior_beta) - torch.sum(torch.lgamma(prior_alpha), dim=1, keepdim=True)
        
        kl = term1 + term2 - term3
        return torch.mean(kl)
    
    def _compute_energy(self, logits, temperature=1.0):
        """改进的能量计算"""
        # 使用logsumexp计算能量分数
        energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
        return energy
    
    def _contrastive_loss(self, features, labels):
        """计算对比损失"""
        if not self.use_contrastive:
            return torch.tensor(0.0).to(features.device)
            
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.contrastive_temp
        
        # 创建标签掩码
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
        mask.fill_diagonal_(False)  # 排除自身
        
        # 正样本掩码
        positive_mask = mask.float()
        
        # 负样本掩码
        negative_mask = (~mask).float()
        
        # 计算对比损失
        neg_logits = torch.exp(similarity_matrix) * negative_mask
        neg_logits = torch.sum(neg_logits, dim=1)
        
        pos_logits = torch.exp(similarity_matrix) * positive_mask
        pos_logits = torch.sum(pos_logits, dim=1)
        
        # 避免除零
        denominator = neg_logits + pos_logits + 1e-8
        
        # 计算损失
        loss = -torch.log(pos_logits / denominator + 1e-8)
        
        # 处理没有正样本的情况
        valid_mask = (torch.sum(positive_mask, dim=1) > 0).float()
        loss = loss * valid_mask
        
        return torch.sum(loss) / (torch.sum(valid_mask) + 1e-8)
    
    def _multi_head_evidence(self, logits):
        """简化的多头不确定性估计"""
        # 不再分割类别，而是使用不同温度参数的多个头
        head_evidences = []
        temperatures = [0.5, 1.0, 2.0]  # 多个温度参数
        
        for temp in temperatures:
            # 每个头使用不同温度参数
            head_evidence = F.softplus(logits / temp)
            head_evidence = torch.clamp(head_evidence, min=1e-6, max=self.max_evidence)
            head_evidences.append(head_evidence)
        
        # 使用学习的权重合并多头evidence
        combined_evidence = torch.zeros_like(head_evidences[0])
        weights = F.softmax(self.head_weights, dim=0)
        
        for i, evidence in enumerate(head_evidences):
            combined_evidence += evidence * weights[i]
        
        return combined_evidence
    
    def _feature_space_regularization(self, features, labels):
        """特征空间正则化"""
        # 计算类内和类间距离
        unique_labels = torch.unique(labels)
        
        if len(unique_labels) <= 1:
            return torch.tensor(0.0).to(features.device)
        
        # 计算类中心
        centers = []
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                center = features[mask].mean(dim=0)
                centers.append(center)
        
        centers = torch.stack(centers)
        
        # 计算类内距离
        intra_distances = []
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() > 0:
                class_features = features[mask]
                center = centers[unique_labels == label].squeeze()
                dist = torch.norm(class_features - center.unsqueeze(0), dim=1).mean()
                intra_distances.append(dist)
        
        intra_distance = torch.stack(intra_distances).mean()
        
        # 计算类间距离
        if len(centers) > 1:
            center_distances = torch.cdist(centers, centers)
            # 移除对角线
            mask = torch.ones_like(center_distances, dtype=torch.bool)
            mask.fill_diagonal_(False)
            inter_distance = center_distances[mask].mean()
        else:
            inter_distance = torch.tensor(0.0).to(features.device)
        
        # 最大化类间距离，最小化类内距离
        return intra_distance - 0.1 * inter_distance
    
    def calculate_evidence_loss(self, logits, features, targets, is_personal=False):
        """计算证据理论损失"""
        # 基础交叉熵损失
        ce_loss = self.ce_loss_fn(logits, targets)
        
        # 多头证据计算
        evidence = self._multi_head_evidence(logits)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # 创建平滑的one-hot标签
        y_one_hot = F.one_hot(targets, self.num_classes).float()
        y_one_hot = y_one_hot * 0.9 + 0.1 / self.num_classes
        
        # 计算EDL对数似然
        log_likelihood = torch.sum(y_one_hot * (torch.digamma(alpha) - torch.digamma(S)), dim=1)
        nll_loss = -log_likelihood.mean()
        
        # 计算能量损失
        energy = self._compute_energy(logits, self.energy_temp)
        energy_loss = torch.mean(torch.abs(energy - self.energy_margin))
        
        # 对比损失
        contrastive_loss = self._contrastive_loss(features, targets)
        
        # 特征空间正则化
        feature_reg_loss = self._feature_space_regularization(features, targets)
        
        # 全局先验KL散度
        kl_div_val = 0.0
        kl_computation_time = 0.0
        
        if is_personal and self.global_prior_alpha is not None:
            prior_alpha = self.global_prior_alpha.unsqueeze(0).expand_as(alpha)
            prior_alpha = torch.clamp(prior_alpha, min=1.0, max=self.max_evidence + 1.0)
            
            start_time = time.time()
            kl_div = self._safe_kl_divergence(alpha, prior_alpha)
            kl_computation_time = time.time() - start_time
            kl_div_val = kl_div
        
        # 组合损失
        total_loss = (
            ce_loss + 
            0.1 * nll_loss +
            0.05 * energy_loss +
            self.contrastive_weight * contrastive_loss +
            self.feature_reg_weight * feature_reg_loss +
            0.5 * kl_div_val
        )
        
        return total_loss, ce_loss, nll_loss, alpha, S, kl_computation_time
    
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
                
                # 前向传播
                logits, features = self.model_global(x, return_features=True)
                
                # 计算损失
                total_loss, _, _, _, _, _ = self.calculate_evidence_loss(logits, features, y, is_personal=False)
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_global.parameters(), max_norm=1.0)
                optimizer.step()
        
        self.model_global.to("cpu")
    
    def _train_personal_model(self, round_idx):
        """训练个性化模型"""
        self.model_personal.load_state_dict(self.model_global.state_dict())
        self.model_personal.train()
        self.model_personal.to(self.device)
        
        global_params = {name: param.clone().detach() for name, param in self.model_global.named_parameters()}
        
        optimizer = torch.optim.SGD(
            self.model_personal.parameters(), 
            lr=self.base_lr * self.personal_lr_factor, 
            momentum=0.9, 
            weight_decay=self.args.wd * 2
        )
        
        acc_meter = AverageMeter()
        kl_time_meter = AverageMeter()
        
        personal_epochs = max(3, self.args.local_epochs // 2)
        train_loader = self.load_train_data()
        if not train_loader: return {'acc': 0, 'kl_div_time': 0}
        
        for epoch in range(personal_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # 前向传播
                logits, features = self.model_personal(x, return_features=True)
                
                # 计算EDL损失
                edl_loss, _, _, _, _, kl_time = self.calculate_evidence_loss(
                    logits, features, y, is_personal=True
                )
                kl_time_meter.update(kl_time)
                
                # 计算近端项
                prox_loss = 0.0
                for name, param in self.model_personal.named_parameters():
                    if name in global_params:
                        prox_loss += ((param - global_params[name].to(self.device)) ** 2).sum()
                
                # 对抗训练
                adv_loss = self._adversarial_training(x, y)
                
                # 总损失
                total_loss = edl_loss + 0.01 * prox_loss + 0.2 * adv_loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_personal.parameters(), max_norm=0.5)
                optimizer.step()
                
                # 更新指标
                with torch.no_grad():
                    pred = F.softmax(logits / self.temperature, dim=1).argmax(1)
                    acc = (pred == y).float().mean() * 100
                    acc_meter.update(acc.item(), x.size(0))
        
        self.model_personal.to("cpu")
        
        return {
            'acc': acc_meter.avg,
            'kl_div_time': kl_time_meter.avg if kl_time_meter.count > 0 else 0.0
        }
    
    def _adversarial_training(self, x, y):
        """改进的对抗训练，更强调OOD检测"""
        self.model_personal.eval()
        x.requires_grad = True
        
        # 计算对抗样本生成的损失
        logits, _ = self.model_personal(x, return_features=True)
        adv_gen_loss = self.ce_loss_fn(logits, y)
        
        # 获取输入梯度
        grad_x = torch.autograd.grad(adv_gen_loss, x, only_inputs=True)[0]
        
        # 创建对抗样本 - 使用更大扰动
        epsilon = 0.1  # 增加扰动强度
        x_adv = x.detach() + epsilon * torch.sign(grad_x.detach())
        x_adv = torch.clamp(x_adv, 0, 1)
        
        # 恢复训练模式
        self.model_personal.train()
        
        # 计算对抗样本的损失
        logits_adv, features_adv = self.model_personal(x_adv, return_features=True)
        
        # 计算能量分数
        energy_adv = self._compute_energy(logits_adv, self.energy_temp)
        
        # 计算对抗样本的不确定性
        evidence_adv = F.softplus(logits_adv / self.temperature)
        alpha_adv = evidence_adv + 1.0
        S_adv = torch.sum(alpha_adv, dim=1)
        epistemic_adv = self.num_classes / S_adv
        
        # 计算熵
        probs_adv = F.softmax(logits_adv / self.temperature, dim=1)
        entropy_adv = -torch.sum(probs_adv * torch.log(probs_adv + 1e-8), dim=1)
        
        # 更强调不确定性最大化
        adv_loss = -(2.0 * entropy_adv.mean() + 1.5 * epistemic_adv.mean()) + 0.01 * energy_adv.mean()
        
        return adv_loss
    
    def train(self, round_idx):
        """训练客户端模型"""
        self.current_round = round_idx
        self._train_global_model()
        stats = self._train_personal_model(round_idx)
        return stats
    
    def get_update(self):
        """获取模型更新"""
        model_weights = deepcopy(self.model_global.state_dict())
        alpha_report = self._compute_alpha_report()
        data_size = self.num_train
        return model_weights, alpha_report, data_size
    
    def _compute_alpha_report(self):
        """计算alpha报告"""
        self.model_global.eval()
        self.model_global.to(self.device)
        
        train_loader = self.load_train_data()
        if not train_loader: return torch.ones(self.num_classes) * 2.0
        
        all_evidence = []
        all_probs = []
        
        with torch.no_grad():
            for x, _ in train_loader:
                x = x.to(self.device)
                logits, _ = self.model_global(x, return_features=True)
                
                # 计算evidence
                evidence = F.softplus(logits / self.temperature)
                all_evidence.append(evidence.cpu())
                
                # 计算概率
                probs = F.softmax(logits / self.temperature, dim=1)
                all_probs.append(probs.cpu())
        
        if not all_evidence: return torch.ones(self.num_classes) * 2.0
        
        # 混合策略：结合evidence和概率
        avg_evidence = torch.cat(all_evidence, dim=0).mean(dim=0)
        avg_probs = torch.cat(all_probs, dim=0).mean(dim=0)
        
        # 使用更保守的先验强度
        prior_strength = min(2.0, 1.0 + 0.01 * self.current_round)
        
        # 混合alpha报告
        alpha_report = 0.5 * (avg_evidence * 0.8 + 1.0) + 0.5 * (avg_probs * prior_strength + 1.0)
        
        self.model_global.to("cpu")
        
        # 更严格的上限，防止过度自信
        return torch.clamp(alpha_report, min=1.1, max=3.0)
    
    def get_eval_output(self, use_personal=True, dataset='test'):
        """获取评估输出"""
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
        all_energies = []
        all_features = []
        
        with torch.no_grad():
            if len(loader) == 0:
                return None
            
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # 前向传播
                logits, features = model_eval(x, return_features=True)
                
                # 计算evidence和alpha
                evidence = F.softplus(logits / self.temperature)
                alpha = evidence + 1.0
                S = torch.sum(alpha, dim=1)
                
                # 计算概率
                probs = alpha / S.unsqueeze(1)
                
                # 计算认知不确定性
                epistemic_uncertainty = self.num_classes / S
                
                # 计算偶然不确定性
                aleatoric_uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                
                # 计算能量分数
                energy = self._compute_energy(logits, self.energy_temp)
                
                # 收集结果
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                all_epistemic_uncertainties.append(epistemic_uncertainty.cpu().numpy())
                all_aleatoric_uncertainties.append(aleatoric_uncertainty.cpu().numpy())
                all_energies.append(energy.cpu().numpy())
                all_features.append(features.cpu().numpy())
        
        model_eval.to("cpu")
        
        if not all_labels:
            return None
        
        return {
            'probs': np.concatenate(all_probs),
            'labels': np.concatenate(all_labels),
            'epistemic_uncertainties': np.concatenate(all_epistemic_uncertainties),
            'aleatoric_uncertainties': np.concatenate(all_aleatoric_uncertainties),
            'energies': np.concatenate(all_energies),
            'features': np.concatenate(all_features)
        }

    def get_update(self):
        model_weights = deepcopy(self.model_global.state_dict())
        alpha_report = self._compute_alpha_report()
        data_size = self.num_train
        return model_weights, alpha_report, data_size
        
    def set_current_global_round(self, current_round):
        self.current_round = current_round

    def _compute_uncertainty(self, logits):
        """简化的不确定性估计方法"""
        # 使用单一温度的softplus激活
        evidence = F.softplus(logits / 2.0)  # 温度参数2.0
        evidence = torch.clamp(evidence, 1e-10, 10.0)  # 限制evidence范围
        
        # 计算Dirichlet参数
        alpha = evidence + 1.0
        
        # 计算不确定性
        S = torch.sum(alpha, dim=1, keepdim=True)
        prob = alpha / S
        
        # 认知不确定性（epistemic）- 使用类别数除以总证据
        epistemic = self.num_classes / torch.sum(alpha, dim=1)
        
        # 偶然不确定性（aleatoric）- 使用预测概率的熵
        aleatoric = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
        
        return prob, alpha, epistemic, aleatoric

    def _compute_edl_loss(self, logits, targets):
        """简化的EDL损失函数"""
        # 计算不确定性
        _, alpha, _, _ = self._compute_uncertainty(logits)
        
        # 计算Dirichlet KL散度损失
        kl_loss = self._dirichlet_kl_divergence(alpha)
        
        # 计算负对数似然损失
        S = torch.sum(alpha, dim=1, keepdim=True)
        target_one_hot = F.one_hot(targets, self.num_classes).float()
        nll_loss = torch.sum(target_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        
        # 总损失
        loss = nll_loss.mean() + 0.01 * kl_loss
        
        return loss

    def train_epoch(self, epoch):
        """简化的训练循环"""
        self.model_personal.train()
        
        train_loss = 0
        train_acc = 0
        train_samples = 0
        train_loader = self.load_train_data()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # 前向传播
            logits, features = self.model_personal(x, return_features=True)
            
            # 计算主要分类损失
            edl_loss = self._compute_edl_loss(logits, y)
            
            # 计算能量损失
            energy_loss = self._energy_based_regularization(logits, x)
            
            # 计算总损失
            loss = edl_loss + 0.1 * energy_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 计算准确率
            pred = torch.argmax(logits, dim=1)
            correct = (pred == y).sum().item()
            
            # 更新统计信息
            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_samples += y.size(0)
        
        return train_loss / train_samples, train_acc / train_samples

    def _energy_based_regularization(self, logits, x):
        """简化的能量正则化"""
        # 计算ID样本的能量
        energy_id = self._compute_energy(logits)
        
        # 生成简单的OOD样本（高斯噪声）
        noise = torch.randn_like(x) * 0.1
        x_noise = torch.clamp(x + noise, 0, 1)
        
        # 计算OOD样本的能量
        logits_noise, _ = self.model_personal(x_noise, return_features=True)
        energy_ood = self._compute_energy(logits_noise)
        
        # 能量损失：ID能量应低，OOD能量应高
        energy_loss = torch.mean(energy_id) - torch.mean(energy_ood) + 10.0
        
        return energy_loss
