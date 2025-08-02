import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader # Assuming Subset might be used in dataset loading
import torchvision.transforms as transforms # For OOD augmentation
import numpy as np
import traceback
from copy import deepcopy
from clients.client_base import Client # Assuming this base class exists
from utils.util import AverageMeter # Assuming this utility exists

# OPTIMIZATION: Increased max_val_for_lgamma and critically fixed term3 for stability
def kl_divergence_dirichlet(alpha, beta, epsilon=1e-8, max_val_for_lgamma=110.0): # MODIFIED: Increased default max_val_for_lgamma
    """
    Calculates KL(Dir(alpha) || Dir(beta)).
    alpha and beta are parameters of Dirichlet distributions (batch_size x num_classes).
    epsilon is for numerical stability.
    max_val_for_lgamma is to prevent lgamma/digamma from exploding with large inputs.
    It should ideally be consistent with or slightly larger than args.clip_alpha_max_val.
    """
    # Clamp inputs to be within a numerically stable range for lgamma/digamma.
    # The primary alpha clipping (using args.clip_alpha_max_val) happens *before* this function.
    # This internal clamping is a safeguard for lgamma/digamma stability.
    safe_alpha = alpha.clamp(min=epsilon, max=max_val_for_lgamma)
    safe_beta = beta.clamp(min=epsilon, max=max_val_for_lgamma)

    # Clamp sums to avoid large inputs into lgamma/digamma for sum_alpha/sum_beta
    max_sum_val = max_val_for_lgamma * alpha.shape[1] # alpha.shape[1] is num_classes

    sum_alpha = torch.sum(safe_alpha, dim=1, keepdim=True).clamp(min=epsilon, max=max_sum_val)
    sum_beta = torch.sum(safe_beta, dim=1, keepdim=True).clamp(min=epsilon, max=max_sum_val)

    term1 = torch.lgamma(sum_alpha) - torch.lgamma(sum_beta)
    term2 = torch.sum(torch.lgamma(safe_beta) - torch.lgamma(safe_alpha), dim=1, keepdim=True)
    
    # MODIFIED: term3 now uses (safe_alpha - safe_beta) for consistency with digamma inputs.
    # This is crucial for stability and correctness of the KL divergence.
    # Added .clamp(min=epsilon) for digamma inputs as an extra safeguard, though safe_alpha/sum_alpha should be positive.
    digamma_safe_alpha = torch.digamma(safe_alpha.clamp(min=epsilon))
    digamma_sum_alpha = torch.digamma(sum_alpha.clamp(min=epsilon)) # sum_alpha is sum of safe_alpha
    term3 = torch.sum((safe_alpha - safe_beta) * (digamma_safe_alpha - digamma_sum_alpha), dim=1, keepdim=True)

    kl_div = term1 + term2 + term3
    
    # Average over batch and ensure KL is non-negative.
    # Clamping to min=0.0 is critical as numerical inaccuracies can sometimes make it slightly negative.
    return kl_div.mean().clamp(min=0.0)


class ClientFedUg(Client):
    def __init__(self, args, client_idx, is_corrupted=False):
        super().__init__(args, client_idx, is_corrupted)
        self.args = args
        # Ditto: 两套模型
        self.model_global = deepcopy(self.model)
        self.model_personal = deepcopy(self.model)
        self.lambda_p = 5.0
        self.local_epochs_personal = 5
        
        # MODIFIED: Use the potentially overridden edl_annealing_epochs from args
        annealing_default_steps = getattr(args, 'edl_annealing_epochs', args.global_rounds / 2.0 if args.global_rounds > 0 else 1.0)
        self.annealing_total_steps_config = annealing_default_steps
        if self.annealing_total_steps_config <= 0:
            self.annealing_total_steps_config = 1.0 # Avoid division by zero or negative

        self.global_prior_alpha_from_server = None
        self.global_model_params_for_prox = None # For FedProx

        if self.args.use_ood_reg:
            self.ood_transform = transforms.Compose([
                transforms.Lambda(lambda x: x + torch.randn_like(x) * self.args.ood_aug_strength),
            ])
        self.current_global_round = 0

    def set_global_prior_alpha(self, global_prior_alpha):
        if global_prior_alpha is not None:
            self.global_prior_alpha_from_server = global_prior_alpha.clone().detach().to(self.device)
        else:
            self.global_prior_alpha_from_server = None
            
    def set_current_global_round(self, current_global_round):
        self.current_global_round = current_global_round

    def set_global_model_params(self, global_model_params):
        if global_model_params:
            self.global_model_params_for_prox = [param.clone().detach().to(self.device) for param in global_model_params]
        else:
            self.global_model_params_for_prox = None

    def calculate_edl_loss(self, evidence_logits, y_one_hot, current_global_round, global_prior_alpha_for_kl, is_personalization_phase=False):
        evidence = self.model.get_evidence(evidence_logits) # Softplus
        alpha = evidence + 1.0

        # --- START: Alpha Clipping (CRITICAL OPTIMIZATION) ---
        clip_max_val = getattr(self.args, 'clip_alpha_max_val', 100.0)
        if clip_max_val <= 1.0:
            if hasattr(self, 'logger') and self.logger:
                self.logger.warning(f"Client {self.client_idx}: clip_alpha_max_val ({clip_max_val}) is <= 1.0. Setting to 100.0 for safety.")
            clip_max_val = 100.0
        alpha = torch.clamp(alpha, min=1e-6, max=clip_max_val)
        # --- END: Alpha Clipping ---

        S_batch = torch.sum(alpha, dim=1, keepdim=True)
        y_one_hot = y_one_hot.float().to(alpha.device)
        safe_alpha_for_loss = alpha.clamp(min=1e-6) 
        safe_S_batch_for_loss = S_batch.clamp(min=1e-6)

        log_likelihood_err = torch.sum(y_one_hot * (torch.digamma(safe_S_batch_for_loss) - torch.digamma(safe_alpha_for_loss)), dim=1, keepdim=True)
        sum_lgamma_alpha = torch.sum(torch.lgamma(safe_alpha_for_loss), dim=1, keepdim=True)
        log_likelihood_var = torch.lgamma(safe_S_batch_for_loss) - sum_lgamma_alpha
        nll_loss = - (log_likelihood_err + log_likelihood_var)
        nll_loss = nll_loss.mean()

        # 改进的KL散度计算
        if self.args.use_global_prior_regularization and global_prior_alpha_for_kl is not None:
            # 对于极端非IID，使用自适应的全局先验
            # 对已见类别使用全局先验，对未见类别使用更强的平滑
            present_classes = y_one_hot.sum(dim=0) > 0
            prior_alpha_expanded = global_prior_alpha_for_kl.unsqueeze(0).expand_as(alpha).clone()
            
            # 对未见类别使用更保守的先验
            unseen_classes = ~present_classes
            if unseen_classes.any():
                # 未见类别使用较小的alpha值，鼓励更高的不确定性
                prior_alpha_expanded[:, unseen_classes] = 1.5  # 接近均匀分布
        else:
            prior_alpha_expanded = torch.ones_like(alpha)
            
        kl_div = kl_divergence_dirichlet(alpha, prior_alpha_expanded)

        # 自适应退火系数
        anneal_coeff = min(1.0, float(current_global_round) / self.annealing_total_steps_config) \
                       if self.annealing_total_steps_config > 0 else 1.0
        
        # 个性化阶段使用不同的KL权重
        if is_personalization_phase:
            lambda_kl = getattr(self.args, 'edl_kl_lambda_personalization', 0.5)
        else:
            lambda_kl = getattr(self.args, 'edl_kl_global_prior_lambda', 1.0)
            
        # === 改进的自适应未见类别KL正则 ===
        present_classes = y_one_hot.sum(dim=0) > 0
        unseen_classes = ~present_classes
        unseen_class_num = unseen_classes.sum().item()
        
        # 对于极端非IID，增强未见类别的正则化
        if unseen_class_num > 0:
            # 动态调整未见类别的权重
            data_imbalance_factor = unseen_class_num / self.num_classes
            lambda_unseen = 2.0 * data_imbalance_factor  # 增强正则化
            
            alpha_unseen = alpha[:, unseen_classes]
            # 使用更强的先验，鼓励未见类别有更高的不确定性
            uniform_prior = torch.ones_like(alpha_unseen) * 1.2  
            kl_unseen = kl_divergence_dirichlet(alpha_unseen, uniform_prior)
            
            # 额外的熵正则化，鼓励未见类别的预测更加平滑
            entropy_unseen = -torch.sum(alpha_unseen / S_batch * torch.log(alpha_unseen / S_batch + 1e-8), dim=1).mean()
            kl_unseen = kl_unseen - 0.1 * entropy_unseen  # 减去熵相当于鼓励更高的熵
        else:
            lambda_unseen = 0.0
            kl_unseen = 0.0

        # 优化的总损失
        total_loss = nll_loss + lambda_kl * anneal_coeff * kl_div + lambda_unseen * kl_unseen
        
        return total_loss, nll_loss, kl_div, alpha, S_batch


    def train(self):
        # 优化1: 两阶段都使用EDL loss，统一训练框架
        self.model_global.train()
        self.model_global.to(self.device)
        optimizer_global = torch.optim.Adam(self.model_global.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        
        # 数据增强
        if hasattr(self.args, 'use_data_augmentation') and self.args.use_data_augmentation:
            transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        else:
            transform_aug = None
            
        # 1. 全局模型训练阶段：使用EDL + CE混合损失
        train_loader = self.load_train_data()
        for epoch in range(self.args.local_epochs):
            if not train_loader:
                continue
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # 数据增强
                if transform_aug and self.model_global.training:
                    x = transform_aug(x)
                
                logits = self.model_global(x)
                ce_loss = F.cross_entropy(logits, y)
                
                # 同时计算EDL loss
                y_one_hot = F.one_hot(y, num_classes=self.num_classes)
                edl_loss, nll, kl_div, alpha, S_batch = self.calculate_edl_loss(
                    logits, y_one_hot, self.current_global_round, 
                    self.global_prior_alpha_from_server, is_personalization_phase=False)
                
                # 混合损失：CE为主，EDL为辅
                total_loss = ce_loss + 0.1 * edl_loss
                
                optimizer_global.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_global.parameters(), 1.0)  # 梯度裁剪
                optimizer_global.step()
                
        self.model_global.to("cpu")
        
        # 2. 个性化阶段：增强的EDL训练
        self.model_personal.load_state_dict(self.model_global.state_dict())
        self.model_personal.train()
        self.model_personal.to(self.device)
        
        # 保存全局模型参数用于正则化
        global_params = {name: param.clone().detach() for name, param in self.model_global.named_parameters()}
        
        # 个性化阶段使用更小的学习率
        lr_personal = getattr(self.args, 'lr_personalization', self.args.lr * 0.1)
        optimizer_personal = torch.optim.Adam(self.model_personal.parameters(), lr=lr_personal, weight_decay=self.args.wd * 2)
        
        total_loss_meter = AverageMeter()
        total_nll_meter = AverageMeter()
        total_kl_meter = AverageMeter()
        total_acc_meter = AverageMeter()
        total_S_avg_meter = AverageMeter()
        
        personal_epochs = getattr(self.args, 'personalization_epochs_on_eval', 5)  # 减少个性化轮数避免过拟合
        
        for epoch in range(personal_epochs):
            train_loader = self.load_train_data()
            if not train_loader:
                continue
                
            # 学习率衰减
            if epoch > personal_epochs // 2:
                for param_group in optimizer_personal.param_groups:
                    param_group['lr'] *= 0.5
                    
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # 强数据增强用于个性化
                if transform_aug:
                    x_aug = transform_aug(x)
                else:
                    x_aug = x + torch.randn_like(x) * 0.02  # 轻微噪声
                
                # Dropout增强
                self.model_personal.train()
                if hasattr(self.model_personal, 'dropout'):
                    self.model_personal.dropout.p = 0.3  # 增加dropout
                
                logits = self.model_personal(x_aug)
                
                # 计算损失
                ce_loss = F.cross_entropy(logits, y)
                y_one_hot = F.one_hot(y, num_classes=self.num_classes)
                
                # 使用全局先验进行个性化
                edl_loss, nll, kl_div, alpha, S_batch = self.calculate_edl_loss(
                    logits, y_one_hot, self.current_global_round, 
                    self.global_prior_alpha_from_server, is_personalization_phase=False)  # 注意这里改为False
                
                # 添加与全局模型的正则化项（类似FedProx）
                prox_term = 0
                for name, param in self.model_personal.named_parameters():
                    if name in global_params:
                        prox_term += ((param - global_params[name].to(self.device)) ** 2).sum()
                
                # 优化的损失函数权重
                lambda_ce = 0.5
                lambda_edl = 0.3  
                lambda_prox = 0.2  
                
                total_loss = lambda_ce * ce_loss + lambda_edl * edl_loss + lambda_prox * prox_term
                
                optimizer_personal.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_personal.parameters(), 1.0)
                optimizer_personal.step()
                
                # 统计
                acc = (logits.argmax(1) == y).float().mean() * 100.0
                total_loss_meter.update(total_loss.item(), x.size(0))
                total_nll_meter.update(nll.item(), x.size(0))
                kl_item = kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div
                total_kl_meter.update(kl_item, x.size(0))
                total_acc_meter.update(acc.item(), x.size(0))
                total_S_avg_meter.update(S_batch.mean().item(), x.size(0))
        
        self.model_personal.to("cpu")
        
        # 返回统计信息
        last_stats = {
            'loss': total_loss_meter.avg if total_loss_meter.count > 0 else float('nan'),
            'acc': total_acc_meter.avg if total_acc_meter.count > 0 else 0.0,
            'nll': total_nll_meter.avg if total_nll_meter.count > 0 else float('nan'),
            'kl': total_kl_meter.avg if total_kl_meter.count > 0 else float('nan'),
            'avg_S': total_S_avg_meter.avg if total_S_avg_meter.count > 0 else self.num_classes,
            'prox_loss': 0.0
        }
        return last_stats

    def get_update(self):
        # 只上传全局模型
        model_weights = deepcopy(self.model_global.state_dict())
        avg_alpha_report = self.compute_average_alpha_report()
        data_size = self.num_train
        return model_weights, avg_alpha_report, data_size

    def evaluate_edl_stats(self, use_personal=True, use_softmax_inference=True):
        # 默认评估个性化模型
        model_eval = self.model_personal if use_personal else self.model_global
        model_eval.eval()
        model_eval.to(self.device)
        log_func_warning = self.logger.warning if hasattr(self, 'logger') and self.logger else print
        log_func_error = self.logger.error if hasattr(self, 'logger') and self.logger else print
        total_loss_meter = AverageMeter()
        total_nll_meter = AverageMeter()
        total_kl_meter = AverageMeter()
        total_acc_meter = AverageMeter()
        total_acc_softmax_meter = AverageMeter()
        total_S_avg_meter = AverageMeter()
        test_loader = self.load_test_data()
        num_classes_val = self.num_classes if hasattr(self, 'num_classes') and self.num_classes > 0 else 10
        if not test_loader or self.num_train < 5:
            log_func_warning(f"Client {self.client_idx}: No test data or too few train samples for EDL stats.")
            return {'test_acc': 0, 'test_loss': float('nan'), 'test_nll': float('nan'), 'test_kl': float('nan'), 'test_avg_S': num_classes_val, 'test_acc_softmax': 0.0}
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                evidence_logits = model_eval(x)
                y_one_hot = F.one_hot(y, num_classes=num_classes_val)
                loss, nll, kl_div, alpha, S_batch = self.calculate_edl_loss(
                    evidence_logits, y_one_hot, self.current_global_round, None, is_personalization_phase=True)
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e6:
                    log_func_error(f"Client {self.client_idx}: NaN/Inf/large loss detected in evaluate_edl_stats. Skipping batch.")
                    continue
                if torch.isnan(kl_div) or torch.isinf(kl_div) or (isinstance(kl_div, torch.Tensor) and kl_div.item() > 1e6):
                    continue
                if torch.isnan(nll) or torch.isinf(nll) or (isinstance(nll, torch.Tensor) and nll.item() > 1e6):
                    continue
                # EDL推理acc
                acc_edl = (alpha.argmax(1) == y).float().mean() * 100.0
                total_acc_meter.update(acc_edl.item(), x.size(0))
                # softmax推理acc
                probs_softmax = F.softmax(evidence_logits, dim=1)
                acc_softmax = (probs_softmax.argmax(1) == y).float().mean() * 100.0
                total_acc_softmax_meter.update(acc_softmax.item(), x.size(0))
                total_loss_meter.update(loss.item(), x.size(0))
                total_nll_meter.update(nll.item(), x.size(0))
                kl_item = kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div
                total_kl_meter.update(kl_item, x.size(0))
                total_S_avg_meter.update(S_batch.mean().item(), x.size(0))
        model_eval.to("cpu")
        eval_stats = {
            'test_acc': total_acc_softmax_meter.avg if use_softmax_inference else total_acc_meter.avg if total_acc_meter.count > 0 else 0.0,
            'test_loss': total_loss_meter.avg if total_loss_meter.count > 0 else float('nan'),
            'test_nll': total_nll_meter.avg if total_nll_meter.count > 0 else float('nan'),
            'test_kl': total_kl_meter.avg if total_kl_meter.count > 0 else float('nan'),
            'test_avg_S': total_S_avg_meter.avg if total_S_avg_meter.count > 0 else num_classes_val,
            'test_acc_softmax': total_acc_softmax_meter.avg if total_acc_softmax_meter.count > 0 else 0.0,
            'test_acc_edl': total_acc_meter.avg if total_acc_meter.count > 0 else 0.0
        }
        return eval_stats

    def compute_average_alpha_report(self):
        self.model.eval()
        self.model.to(self.device)
        log_func_warning = self.logger.warning if hasattr(self, 'logger') and self.logger else print
        
        all_alphas = []
        data_loader = self.load_train_data()

        # 统计本地类别数
        label_set = set()
        if data_loader:
            for _, y in data_loader:
                label_set.update(y.cpu().numpy().tolist())
        num_unique_labels = len(label_set)

        if not data_loader or num_unique_labels < 2 or self.num_train < 5:
            log_func_warning(f"Client {self.client_idx}: Too few classes ({num_unique_labels}) or samples ({self.num_train}) for alpha report, returning uniform.")
            default_alpha_val = getattr(self.args, 'clip_alpha_max_val', 100.0) / (self.num_classes if hasattr(self, 'num_classes') and self.num_classes > 0 else 10.0)
            return torch.ones(self.num_classes if hasattr(self, 'num_classes') else 10) * default_alpha_val

        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(self.device)
                evidence_logits = self.model(x)
                evidence = self.model.get_evidence(evidence_logits)
                alpha = evidence + 1.0
                
                clip_max_val = getattr(self.args, 'clip_alpha_max_val', 100.0)
                if clip_max_val <= 1.0: clip_max_val = 100.0 # Safety check
                alpha = torch.clamp(alpha, min=1e-6, max=clip_max_val) 
                
                all_alphas.append(alpha.cpu())
        
        self.model.to("cpu")

        if not all_alphas:
            log_func_warning(f"Client {self.client_idx}: Alpha list empty, returning default for alpha report.")
            default_alpha_val = getattr(self.args, 'clip_alpha_max_val', 100.0) / (self.num_classes if hasattr(self, 'num_classes') and self.num_classes > 0 else 10.0)
            return torch.ones(self.num_classes if hasattr(self, 'num_classes') else 10) * default_alpha_val

        avg_alpha_report = torch.cat(all_alphas, dim=0).mean(dim=0)
        # Dirichlet smoothing
        lam = 0.1
        uniform = torch.ones_like(avg_alpha_report) * avg_alpha_report.mean()
        avg_alpha_report = (1-lam)*avg_alpha_report + lam*uniform
        return torch.clamp(avg_alpha_report, min=1e-6) 

    def train_ce_only(self):
        # 只用交叉熵loss训练EDL模型
        self.model_global.train()
        self.model_global.to(self.device)
        optimizer = torch.optim.Adam(self.model_global.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        ce_loss_fn = torch.nn.CrossEntropyLoss()
        total_loss_meter = AverageMeter()
        total_acc_meter = AverageMeter()
        train_loader = self.load_train_data()
        if not train_loader:
            return {'loss': float('nan'), 'acc': 0.0}
        for epoch in range(self.args.local_epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model_global(x)
                loss = ce_loss_fn(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = (logits.argmax(1) == y).float().mean() * 100.0
                total_loss_meter.update(loss.item(), x.size(0))
                total_acc_meter.update(acc.item(), x.size(0))
        self.model_global.to("cpu")
        return {'loss': total_loss_meter.avg if total_loss_meter.count > 0 else float('nan'),
                'acc': total_acc_meter.avg if total_acc_meter.count > 0 else 0.0} 