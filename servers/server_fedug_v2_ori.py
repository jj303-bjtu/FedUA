import torch
import numpy as np
import time
from copy import deepcopy
from servers.server_base import Server
from clients.client_fedug_v2 import ClientFedUgV2
from utils.util import AverageMeter

class ServerFedUgV2(Server):
    """
    重构的FedUG服务器，专注于稳定性和泛化性
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        # 创建客户端
        self.clients = []
        for i in range(args.num_clients):
            client = ClientFedUgV2(args, i)
            self.clients.append(client)
        
        # 全局先验初始化为保守值
        self.global_prior_alpha = torch.ones(self.num_classes) * 2.0
        
        # 性能记录
        self.round_times = []
        self.train_times = []
        
        print(f"Initialized FedUG V2 Server with {len(self.clients)} clients")
    
    def send_models(self):
        """发送全局模型和先验给客户端"""
        for client in self.clients:
            client.model.load_state_dict(self.model.state_dict())
            client.model_global.load_state_dict(self.model.state_dict())
            client.set_global_prior_alpha(self.global_prior_alpha)
    
    def train_round(self, round_num):
        """单轮训练"""
        self.logger.info(f"--- Global Round {round_num}/{self.global_rounds} ---")
        self.sample_active_clients()
        # 采样客户端
        active_clients = self.active_clients
        self.logger.info(f"Selected {len(active_clients)} active clients for round {round_num}.")
        
        # 发送模型
        self.send_models()
        
        # 客户端训练
        client_updates = []
        client_alpha_reports = []
        client_data_sizes = []
        
        train_stats = AverageMeter()
        
        for i, client in enumerate(active_clients):
            self.logger.info(f"Training client {client.client_idx} ({i+1}/{len(active_clients)})...")
            
            try:
                # 设置当前轮次
                client.set_current_global_round(round_num)
                
                # 客户端训练
                stats = client.train()
                
                # 获取更新
                model_weights, alpha_report, data_size = client.get_update()
                
                client_updates.append(model_weights)
                client_alpha_reports.append(alpha_report)
                client_data_sizes.append(data_size)
                
                # 记录统计
                train_stats.update(stats['acc'], data_size)
                
                self.logger.info(f"Client {client.client_idx}: Acc={stats['acc']:.2f}, "
                                 f"Loss={stats['loss']:.4f}, CE={stats['ce_loss']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training client {client.client_idx}: {str(e)}")
                continue
        
        # 聚合模型
        if client_updates:
            self.aggregate_models(client_updates, client_data_sizes)
            self.aggregate_alpha_reports(client_alpha_reports, client_data_sizes)
        
        # 记录轮次统计
        avg_train_acc = train_stats.avg if train_stats.count > 0 else 0.0
        self.logger.info(f"Round [{round_num}/{self.global_rounds}] "
                         f"Avg Train Acc: {avg_train_acc:.2f}%")
        
        return avg_train_acc, 0.0  # 返回准确率和损失
    
    def aggregate_models(self, client_updates, client_data_sizes):
        """简化的模型聚合"""
        if not client_updates:
            return
        
        total_data_size = sum(client_data_sizes)
        if total_data_size == 0:
            return
        
        # 初始化聚合参数 - 确保在正确的设备上
        global_state_dict = self.model.state_dict()
        aggregated_state_dict = {}
        
        for name in global_state_dict.keys():
            aggregated_state_dict[name] = torch.zeros_like(global_state_dict[name]).to(self.device)
        
        # 加权聚合
        for i, client_state_dict in enumerate(client_updates):
            weight = client_data_sizes[i] / total_data_size
            
            for name in aggregated_state_dict.keys():
                if name in client_state_dict:
                    # 跳过BatchNorm的统计量
                    if 'num_batches_tracked' in name:
                        if i == 0:  # 使用第一个客户端的值
                            aggregated_state_dict[name] = client_state_dict[name].to(self.device)
                        continue
                    
                    param = client_state_dict[name].to(self.device)
                    aggregated_state_dict[name] += weight * param
        
        # 将聚合后的参数移回CPU再更新模型（如果模型在CPU上）
        final_state_dict = {}
        for name, param in aggregated_state_dict.items():
            final_state_dict[name] = param.cpu()
        
        # 更新全局模型
        self.model.load_state_dict(final_state_dict)
    
    def aggregate_alpha_reports(self, alpha_reports, client_data_sizes):
        """简化的先验聚合"""
        if not alpha_reports:
            return
        
        valid_reports = []
        valid_sizes = []
        
        for report, size in zip(alpha_reports, client_data_sizes):
            if isinstance(report, torch.Tensor) and report.numel() == self.num_classes:
                valid_reports.append(report)
                valid_sizes.append(size)
        
        if not valid_reports:
            return
        
        # 加权平均
        total_size = sum(valid_sizes)
        if total_size == 0:
            return
        
        weighted_sum = torch.zeros_like(valid_reports[0])
        for report, size in zip(valid_reports, valid_sizes):
            weight = size / total_size
            weighted_sum += weight * report.to(weighted_sum.device)
        
        # 平滑处理，避免极端值
        smoothing_factor = 0.1
        uniform_prior = torch.ones_like(weighted_sum) * 2.0
        self.global_prior_alpha = (1 - smoothing_factor) * weighted_sum + smoothing_factor * uniform_prior
        
        # 确保合理范围
        self.global_prior_alpha = torch.clamp(self.global_prior_alpha, min=1.1, max=5.0)
    
    def evaluate_models(self, current_round):
        """评估模型性能"""
        self.logger.info(f"--- Evaluating models at round {current_round} ---")
        
        # 全局模型评估
        global_acc_meter = AverageMeter()
        global_loss_meter = AverageMeter()
        
        # 个性化模型评估
        personal_acc_meter = AverageMeter()
        personal_loss_meter = AverageMeter()
        
        for client in self.clients:
            try:
                # 全局模型评估
                global_stats = client.evaluate_edl_stats(use_personal=False)
                global_acc_meter.update(global_stats['test_acc'], client.num_test)
                if not np.isinf(global_stats['test_loss']):
                    global_loss_meter.update(global_stats['test_loss'], client.num_test)
                
                # 个性化模型评估
                personal_stats = client.evaluate_edl_stats(use_personal=True)
                personal_acc_meter.update(personal_stats['test_acc'], client.num_test)
                if not np.isinf(personal_stats['test_loss']):
                    personal_loss_meter.update(personal_stats['test_loss'], client.num_test)
                
            except Exception as e:
                self.logger.warning(f"Error evaluating client {client.client_idx}: {str(e)}")
                continue
        
        # 输出结果
        self.logger.info(f"Global Model Evaluation (Round {current_round}):")
        self.logger.info(f"  Avg Test Acc: {global_acc_meter.avg:.2f}%")
        self.logger.info(f"  Avg Test Loss: {global_loss_meter.avg:.4f}")
        
        self.logger.info(f"Personalized Model Evaluation (Round {current_round}):")
        self.logger.info(f"  Avg Test Acc: {personal_acc_meter.avg:.2f}%")
        self.logger.info(f"  Avg Test Loss: {personal_loss_meter.avg:.4f}")
        
        return personal_acc_meter.avg, personal_loss_meter.avg
    
    def train(self):
        """主训练循环"""
        self.logger.info("Starting FedUG V2 Training Process")
        
        best_acc = 0.0
        
        for round_num in range(1, self.global_rounds + 1):
            start_time = time.time()
            
            # 训练一轮
            train_acc, train_loss = self.train_round(round_num)
            
            round_time = time.time() - start_time
            self.round_times.append(round_time)
            
            # 评估
            if round_num % self.args.eval_gap == 0 or round_num == self.global_rounds:
                test_acc, test_loss = self.evaluate_models(round_num)
                
                # 记录最佳性能
                if test_acc > best_acc:
                    best_acc = test_acc
                    self.logger.info(f"New best personalized accuracy: {best_acc:.2f}%")
            
            self.logger.info(f"Round [{round_num}] completed in {round_time:.2f}s")
        
        self.logger.info(f"Training completed. Best personalized accuracy: {best_acc:.2f}%")
        return best_acc 