# In client_fedfope.py

from utils.util import AverageMeter # 假设 AverageMeter 在此路径
import torch
from clients.client_base import Client # 假设 Client 基类在此路径
from copy import deepcopy
import logging # 引入 logging

def get_model_params_vector(model, device='cpu'):
    """Flattens all model parameters that require gradients into a single 1D tensor."""
    params = []
    for param in model.parameters():
        if param.requires_grad: # 确保只包含需要梯度的参数
            params.append(param.data.view(-1).to(device))
    return torch.cat(params) if params else torch.empty(0, device=device)

def fourier_transform_update(update_vector):
    """Transforms the model update vector to the Fourier domain using torch.fft."""
    return torch.fft.fft(update_vector.float())

class ClientFedFope(Client):
    def __init__(self, args, client_idx):
        super().__init__(args, client_idx)
        self.args = args
        self.latest_params = None
        self.initial_global_params_vector = None

        # 初始化 logger
        logger_name = f"ClientFedFope_{client_idx}"
        self.logger = logging.getLogger(logger_name)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


    def set_model(self, global_model_instance):
        """Receives the global model from the server and stores its initial state."""
        self.model = deepcopy(global_model_instance).to(self.device)
        # 确保在提取初始参数前模型已在正确设备上
        self.initial_global_params_vector = get_model_params_vector(self.model.to(self.device), device=self.device).clone().detach()


    def train(self):
        trainloader = self.load_train_data()
        self.model = self.model.to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)

        self.model.train()
        losses = AverageMeter()
        accs = AverageMeter()

        for e in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss_val = self.loss(output, y) # 使用 self.loss

                acc = (output.argmax(1) == y).float().mean() * 100.0
                accs.update(acc.item(), x.size(0))
                losses.update(loss_val.item(), x.size(0))

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

        self.model = self.model.to("cpu")
        self.latest_params = deepcopy(self.model.state_dict())
        
        self.logger.info(f"Client {self.client_idx} training complete. Avg Acc: {accs.avg:.2f}%, Avg Loss: {losses.avg:.4f}")
        return accs.avg, losses.avg

    def get_fourier_transformed_update(self):
        """Calculates model delta, optionally clips it, transforms it to Fourier domain, and returns."""
        if self.model is None or self.initial_global_params_vector is None:
            self.logger.error(f"Client {self.client_idx} model or initial_params not set for delta calculation.")
            return None

        self.model.eval().to(self.device) # 确保模型在正确设备上进行参数提取
        trained_local_params_vector = get_model_params_vector(self.model, device=self.device)
        
        # 确保 initial_global_params_vector 也在同一设备上
        model_delta_vector = trained_local_params_vector - self.initial_global_params_vector.to(self.device)

        # --- Client-Side Delta Clipping ---
        delta_clip_threshold = getattr(self.args, 'delta_clip_threshold', 0.0) # 从args获取，0.0表示不裁剪
        if delta_clip_threshold > 0:
            delta_norm = torch.norm(model_delta_vector)
            if delta_norm > delta_clip_threshold:
                self.logger.debug(f"Client {self.client_idx}: Clipping delta norm from {delta_norm:.4f} to {delta_clip_threshold:.4f}")
                model_delta_vector = model_delta_vector * (delta_clip_threshold / delta_norm)
        
        if torch.isnan(model_delta_vector).any() or torch.isinf(model_delta_vector).any():
            self.logger.warning(f"Client {self.client_idx}: NaN/Inf detected in delta after potential clipping. Replacing with zeros.")
            model_delta_vector = torch.zeros_like(model_delta_vector)
            # Consider returning None to let server skip this update:
            # return None

        fourier_transformed_delta = fourier_transform_update(model_delta_vector)
        return fourier_transformed_delta.cpu() # 将结果移回CPU，减少GPU占用

    def perform_local_training_and_get_fourier_update(self):
        """Main method called by server: trains locally and returns Fourier update."""
        avg_acc, avg_loss = self.train() # train已经包含了 latest_params 的更新
        fourier_update = self.get_fourier_transformed_update()
        num_samples = self.num_train
        self.logger.debug(f"Client {self.client_idx} sending update. Samples: {num_samples}, Acc: {avg_acc:.2f}%, Loss: {avg_loss:.4f}")
        return fourier_update, avg_acc, avg_loss, num_samples

    # get_update 方法可以保留，以防FedAvgFT等场景需要（尽管FedFope不直接用它发送更新给服务器）
    def get_update(self, global_model_state_dict):
        """
        Calculates the update vector (difference based on state_dicts).
        This is more aligned with how FedAvg client might calculate delta if needed for other purposes.
        FedFope uses get_fourier_transformed_update for its primary mechanism.
        """
        update = {}
        self.model.eval() # Ensure model is in eval mode
        # self.latest_params should contain the state_dict after local training on CPU
        local_params = self.latest_params 
        if local_params is None:
            self.logger.error(f"Client {self.client_idx}: latest_params not set. Cannot compute update via get_update.")
            return None # Or an empty dict

        for name, global_param_tensor in global_model_state_dict.items():
            if name in local_params:
                local_param_tensor = local_params[name].to("cpu")
                delta = local_param_tensor - global_param_tensor.to("cpu")
                if torch.isnan(delta).any() or torch.isinf(delta).any():
                    self.logger.warning(f"Warning: NaN/Inf detected in delta for param {name} in client {self.client_idx} (get_update). Appending zeros.")
                    delta = torch.zeros_like(delta)
                update[name] = delta
            else:
                self.logger.warning(f"Warning: Parameter {name} not found in local model during get_update for client {self.client_idx}.")
        return update