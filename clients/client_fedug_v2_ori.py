import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from copy import deepcopy
from clients.client_base import Client
from utils.util import AverageMeter

class ClientFedUgV2(Client):
    """
    彻底重构的FedUG客户端，解决过拟合和泛化性差的问题
    """
    def __init__(self, args, client_idx, is_corrupted=False):
        super().__init__(args, client_idx, is_corrupted)
        self.args = args
        
        # 使用轻量化的双模型架构
        self.model_global = deepcopy(self.model)
        self.model_personal = deepcopy(self.model) 
        
        # 关键参数调整
        self.uncertainty_threshold = 2.0  # 降低不确定性阈值
        self.max_evidence = 5.0  # 严格限制evidence上限
        self.edl_weight = 0.1  # 降低EDL权重避免过拟合
        self.global_consistency_weight = 1.0  # 增强全局一致性
        
        # 温度缩放用于校准
        self.temperature = 2.0
        
        # 全局先验和当前轮次
        self.global_prior_alpha = None
        self.current_round = 0
        
        # 自适应学习率
        self.base_lr = args.lr
        self.personal_lr_factor = 0.1
        
    def set_global_prior_alpha(self, global_prior_alpha):
        if global_prior_alpha is not None:
            self.global_prior_alpha = global_prior_alpha.clone().detach().to(self.device)
    
    def set_current_global_round(self, current_round):
        self.current_round = current_round
    
    def calculate_evidence_loss(self, logits, targets, is_personal=False):
        """
        重新设计的EDL损失函数，避免负损失和过度自信
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        
        # 1. 通过温度缩放calibration
        calibrated_logits = logits / self.temperature
        
        # 2. 计算evidence，严格限制上界
        evidence = F.softplus(calibrated_logits)
        evidence = torch.clamp(evidence, min=1e-6, max=self.max_evidence)
        
        # 3. Dirichlet参数
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=1, keepdim=True)
        
        # 4. 计算标准交叉熵损失作为主要监督信号
        ce_loss = F.cross_entropy(calibrated_logits, targets)
        
        # 5. EDL不确定性正则化
        y_one_hot = F.one_hot(targets, num_classes).float()
        
        # 正确的Dirichlet-Multinomial likelihood
        expected_prob = alpha / S
        digamma_sum = torch.digamma(S)
        digamma_alpha = torch.digamma(alpha)
        
        # 正确的对数似然
        log_likelihood = torch.sum(y_one_hot * (digamma_alpha - digamma_sum), dim=1)
        nll_loss = -log_likelihood.mean()
        
        # 6. 不确定性正则化：鼓励对未见数据保持不确定性
        present_classes = y_one_hot.sum(dim=0) > 0
        
        # 对已见类别：降低不确定性
        seen_alpha = alpha[:, present_classes]
        seen_uncertainty = num_classes / torch.sum(seen_alpha, dim=1)
        uncertainty_loss_seen = seen_uncertainty.mean()
        
        # 对未见类别：保持适度不确定性
        if present_classes.sum() < num_classes:
            unseen_mask = ~present_classes
            unseen_alpha = alpha[:, unseen_mask]
            target_uncertainty = self.uncertainty_threshold
            unseen_uncertainty = torch.abs(
                (unseen_mask.sum().float() / torch.sum(unseen_alpha, dim=1)) - target_uncertainty
            )
            uncertainty_loss_unseen = unseen_uncertainty.mean()
        else:
            uncertainty_loss_unseen = 0.0
        
        # 7. 全局一致性损失（仅在个性化阶段）
        global_consistency_loss = 0.0
        if is_personal and self.global_prior_alpha is not None:
            # KL散度：q(α_personal) || p(α_global)
            prior_alpha = self.global_prior_alpha.unsqueeze(0).expand_as(alpha)
            prior_alpha = torch.clamp(prior_alpha, min=1.0, max=self.max_evidence + 1.0)
            
            kl_div = self._safe_kl_divergence(alpha, prior_alpha)
            global_consistency_loss = kl_div
        
        # 8. 组合最终损失
        total_loss = (
            ce_loss + 
            self.edl_weight * nll_loss +
            0.1 * uncertainty_loss_seen +
            0.05 * uncertainty_loss_unseen +
            self.global_consistency_weight * global_consistency_loss * (0.5 if is_personal else 0.0)
        )
        
        return total_loss, ce_loss, nll_loss, alpha, S
    
    def _safe_kl_divergence(self, alpha1, alpha2):
        """安全的KL散度计算"""
        # 使用Beta函数的对数来计算KL散度
        sum_alpha1 = torch.sum(alpha1, dim=1)
        sum_alpha2 = torch.sum(alpha2, dim=1)
        
        # 避免数值不稳定
        alpha1_safe = torch.clamp(alpha1, min=1e-6, max=100.0)
        alpha2_safe = torch.clamp(alpha2, min=1e-6, max=100.0)
        sum_alpha1_safe = torch.clamp(sum_alpha1, min=1e-6, max=1000.0)
        sum_alpha2_safe = torch.clamp(sum_alpha2, min=1e-6, max=1000.0)
        
        term1 = torch.lgamma(sum_alpha1_safe) - torch.lgamma(sum_alpha2_safe)
        term2 = torch.sum(torch.lgamma(alpha2_safe) - torch.lgamma(alpha1_safe), dim=1)
        term3 = torch.sum((alpha1_safe - alpha2_safe) * 
                         (torch.digamma(alpha1_safe) - torch.digamma(sum_alpha1_safe.unsqueeze(1))), dim=1)
        
        kl = term1 + term2 + term3
        return torch.clamp(kl, min=0.0).mean()
    
    def train(self):
        """重构的训练流程"""
        # 第一阶段：全局模型训练（主要用CE loss）
        self._train_global_model()
        
        # 第二阶段：个性化训练（CE + 轻量EDL）
        stats = self._train_personal_model()
        
        return stats
    
    def _train_global_model(self):
        """全局模型训练阶段"""
        self.model_global.train()
        self.model_global.to(self.device)
        
        optimizer = torch.optim.SGD(
            self.model_global.parameters(), 
            lr=self.base_lr, 
            momentum=0.9, 
            weight_decay=self.args.wd
        )
        
        # 数据增强
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ])
        
        for epoch in range(self.args.local_epochs):
            train_loader = self.load_train_data()
            if not train_loader:
                continue
                
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                # 轻微数据增强
                if np.random.random() < 0.3:
                    x = transform_train(x)
                
                # 前向传播
                logits = self.model_global(x)
                
                # 混合损失：主要用CE，轻微EDL正则化
                ce_loss = F.cross_entropy(logits, y)
                edl_loss, _, _, _, _ = self.calculate_evidence_loss(logits, y, is_personal=False)
                
                total_loss = 0.9 * ce_loss + 0.1 * edl_loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model_global.parameters(), max_norm=1.0)
                
                optimizer.step()
        
        self.model_global.to("cpu")
    
    def _train_personal_model(self):
        """个性化模型训练阶段"""
        # 初始化为全局模型
        self.model_personal.load_state_dict(self.model_global.state_dict())
        self.model_personal.train()
        self.model_personal.to(self.device)
        
        # 保存全局模型参数用于正则化
        global_params = {name: param.clone().detach() 
                        for name, param in self.model_global.named_parameters()}
        
        # 个性化优化器：更小的学习率
        personal_lr = self.base_lr * self.personal_lr_factor
        optimizer = torch.optim.SGD(
            self.model_personal.parameters(),
            lr=personal_lr,
            momentum=0.9,
            weight_decay=self.args.wd * 2
        )
        
        # 统计
        total_loss_meter = AverageMeter()
        ce_loss_meter = AverageMeter()
        nll_loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        # 较少的个性化轮数避免过拟合
        personal_epochs = max(3, self.args.local_epochs // 2)
        
        for epoch in range(personal_epochs):
            train_loader = self.load_train_data()
            if not train_loader:
                continue
            
            # 学习率衰减
            if epoch > personal_epochs // 2:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # 前向传播
                logits = self.model_personal(x)
                
                # EDL损失
                edl_loss, ce_loss, nll_loss, alpha, S = self.calculate_evidence_loss(
                    logits, y, is_personal=True
                )
                
                # FedProx正则化
                prox_loss = 0.0
                for name, param in self.model_personal.named_parameters():
                    if name in global_params:
                        prox_loss += ((param - global_params[name].to(self.device)) ** 2).sum()
                
                # 总损失
                total_loss = edl_loss + 0.01 * prox_loss
                
                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_personal.parameters(), max_norm=0.5)
                optimizer.step()
                
                # 统计
                with torch.no_grad():
                    pred = F.softmax(logits / self.temperature, dim=1).argmax(1)
                    acc = (pred == y).float().mean() * 100
                    
                    total_loss_meter.update(total_loss.item(), x.size(0))
                    ce_loss_meter.update(ce_loss.item(), x.size(0))
                    nll_loss_meter.update(nll_loss.item(), x.size(0))
                    acc_meter.update(acc.item(), x.size(0))
        
        self.model_personal.to("cpu")
        
        return {
            'loss': total_loss_meter.avg,
            'ce_loss': ce_loss_meter.avg,
            'nll_loss': nll_loss_meter.avg,
            'acc': acc_meter.avg,
            'avg_S': 0.0,  # 不再关注过度饱和的S值
            'prox_loss': 0.0
        }
    
    def get_update(self):
        """返回全局模型权重和简化的先验"""
        model_weights = deepcopy(self.model_global.state_dict())
        alpha_report = self._compute_alpha_report()
        data_size = self.num_train
        return model_weights, alpha_report, data_size
    
    def _compute_alpha_report(self):
        """计算更稳定的先验报告"""
        self.model_global.eval()
        self.model_global.to(self.device)
        
        all_probs = []
        train_loader = self.load_train_data()
        
        if not train_loader:
            # 返回保守的均匀先验
            return torch.ones(self.num_classes) * 2.0
        
        with torch.no_grad():
            for x, _ in train_loader:
                x = x.to(self.device)
                logits = self.model_global(x)
                
                # 使用softmax概率而不是evidence
                probs = F.softmax(logits / self.temperature, dim=1)
                all_probs.append(probs.cpu())
        
        self.model_global.to("cpu")
        
        if not all_probs:
            return torch.ones(self.num_classes) * 2.0
        
        # 计算平均概率，转换为保守的Dirichlet参数
        avg_probs = torch.cat(all_probs, dim=0).mean(dim=0)
        
        # 转换为合理的先验强度（避免过度自信）
        prior_strength = 5.0  # 保守的先验强度
        alpha_report = avg_probs * prior_strength + 1.0
        
        return torch.clamp(alpha_report, min=1.1, max=3.0)
    
    def evaluate_edl_stats(self, use_personal=True, use_softmax_inference=True):
        """评估统计，使用校准后的预测"""
        model_eval = self.model_personal if use_personal else self.model_global
        model_eval.eval()
        model_eval.to(self.device)
        
        acc_meter = AverageMeter()
        loss_meter = AverageMeter()
        test_loader = self.load_test_data()
        
        if not test_loader:
            return {
                'test_acc': 0.0,
                'test_loss': float('inf'),
                'test_nll': float('inf'),
                'test_kl': 0.0,
                'test_avg_S': self.num_classes,
                'test_acc_softmax': 0.0,
                'test_acc_edl': 0.0
            }
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                logits = model_eval(x)
                
                # 使用温度缩放的校准预测
                calibrated_logits = logits / self.temperature
                probs = F.softmax(calibrated_logits, dim=1)
                pred = probs.argmax(1)
                
                acc = (pred == y).float().mean() * 100
                loss = F.cross_entropy(calibrated_logits, y)
                
                acc_meter.update(acc.item(), x.size(0))
                loss_meter.update(loss.item(), x.size(0))
        
        model_eval.to("cpu")
        
        return {
            'test_acc': acc_meter.avg,
            'test_loss': loss_meter.avg,
            'test_nll': loss_meter.avg,
            'test_kl': 0.0,
            'test_avg_S': self.num_classes,
            'test_acc_softmax': acc_meter.avg,
            'test_acc_edl': acc_meter.avg
        } 