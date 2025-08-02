import torch
import torch.nn.functional as F
import numpy as np
from utils.util import AverageMeter
from clients.client_base import Client

class ClientFedAvg(Client):
    def __init__(self, args, client_idx, is_corrupted=False):
        super().__init__(args, client_idx, is_corrupted)        
        
    def train(self):
        trainloader = self.load_train_data()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
        self.model = self.model.to(self.device)
        self.model.train()
        losses = AverageMeter()
        accs = AverageMeter()

        for e in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)

                # forward pass
                output = self.model(x)
                loss = self.loss(output, y)

                acc = (output.argmax(1) == y).float().mean() * 100.0
                accs.update(acc, x.size(0))
                losses.update(loss.item(), x.size(0))

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        self.model = self.model.to("cpu")
        return accs.avg, losses.avg

    def get_eval_output(self):
        """
        在测试(ID)和OOD数据上评估模型，以获取用于服务器端高级指标计算的输出。
        """
        self.model.eval()
        self.model.to(self.device)

        # --- In-Distribution (ID) 评估 ---
        id_loader = self.load_test_data()
        all_id_probs, all_id_labels, all_id_uncertainties = [], [], []

        if id_loader and len(id_loader) > 0:
            with torch.no_grad():
                for x, y in id_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.model(x)
                    probs = F.softmax(output, dim=1)
                    
                    # 基于最大softmax概率(MSP)的不确定性
                    msp, _ = torch.max(probs, dim=1)
                    uncertainty = 1 - msp

                    all_id_probs.append(probs.cpu().numpy())
                    all_id_labels.append(y.cpu().numpy())
                    all_id_uncertainties.append(uncertainty.cpu().numpy())

        # --- Out-of-Distribution (OOD) 评估 ---
        ood_loader = self.load_ood_data()
        all_ood_uncertainties = []

        if ood_loader and len(ood_loader) > 0:
            with torch.no_grad():
                for x, y in ood_loader: # OOD标签不用于检测
                    x = x.to(self.device)
                    output = self.model(x)
                    probs = F.softmax(output, dim=1)

                    # 基于最大softmax概率(MSP)的不确定性
                    msp, _ = torch.max(probs, dim=1)
                    uncertainty = 1 - msp
                    
                    all_ood_uncertainties.append(uncertainty.cpu().numpy())

        self.model.to("cpu")

        # 聚合结果
        results = {
            'id_probs': np.concatenate(all_id_probs) if all_id_probs else np.array([]),
            'id_labels': np.concatenate(all_id_labels) if all_id_labels else np.array([]),
            'id_uncertainties': np.concatenate(all_id_uncertainties) if all_id_uncertainties else np.array([]),
            'ood_uncertainties': np.concatenate(all_ood_uncertainties) if all_ood_uncertainties else np.array([])
        }
        return results