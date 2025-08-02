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

# Fourier transform is now primarily a server concern if GFT is used,
# but FFT can be a fallback. Client will send raw delta.
# def fourier_transform_update(update_vector):
#     """Transforms the model update vector to the Fourier domain using torch.fft."""
#     return torch.fft.fft(update_vector.float())

class ClientFedFope(Client):
    def __init__(self, args, client_idx):
        super().__init__(args, client_idx)
        self.args = args
        # self.latest_params = None # latest_params (state_dict) is managed by base or if needed for get_update
        self.initial_global_params_vector = None

        # 初始化 logger
        logger_name = f"ClientFedFope_{client_idx}"
        self.logger = logging.getLogger(logger_name)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(getattr(logging, "INFO", logging.INFO))


    def set_model(self, global_model_instance):
        """Receives the global model from the server and stores its initial state."""
        self.model = deepcopy(global_model_instance).to(self.device)
        # 确保在提取初始参数前模型已在正确设备上
        self.initial_global_params_vector = get_model_params_vector(self.model.to(self.device), device=self.device).clone().detach()


    def train(self):
        trainloader = self.load_train_data()
        self.model = self.model.to(self.device)
        # Ensure self.lr, self.momentum, self.wd are attributes of ClientFedFope or args
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)

        self.model.train()
        losses = AverageMeter()
        accs = AverageMeter()

        for e in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                # Ensure self.loss is a callable loss function
                loss_val = self.loss(output, y) 

                acc = (output.argmax(1) == y).float().mean() * 100.0
                accs.update(acc.item(), x.size(0))
                losses.update(loss_val.item(), x.size(0))

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

        # self.model is already on self.device, move to CPU after getting params vector if needed
        # self.latest_params = deepcopy(self.model.state_dict()) # Keep state_dict if get_update is used
        
        self.logger.info(f"Client {self.client_idx} training complete. Avg Acc: {accs.avg:.2f}%, Avg Loss: {losses.avg:.4f}")
        return accs.avg, losses.avg

    def get_model_delta_vector(self):
        """Calculates model delta (trained_params - initial_params) and optionally clips it."""
        if self.model is None or self.initial_global_params_vector is None:
            self.logger.error(f"Client {self.client_idx} model or initial_params not set for delta calculation.")
            return None

        self.model.eval().to(self.device) # 确保模型在正确设备上进行参数提取
        trained_local_params_vector = get_model_params_vector(self.model, device=self.device)
        
        # 确保 initial_global_params_vector 也在同一设备上
        model_delta_vector = trained_local_params_vector - self.initial_global_params_vector.to(self.device)

        # --- Client-Side Delta Clipping ---
        # Ensure delta_clip_threshold is an attribute of self.args
        delta_clip_threshold = getattr(self.args, 'delta_clip_threshold', 0.0) 
        if delta_clip_threshold > 0:
            delta_norm = torch.norm(model_delta_vector)
            if delta_norm > delta_clip_threshold:
                self.logger.debug(f"Client {self.client_idx}: Clipping delta norm from {delta_norm:.4f} to {delta_clip_threshold:.4f}")
                model_delta_vector = model_delta_vector * (delta_clip_threshold / delta_norm)
        
        if torch.isnan(model_delta_vector).any() or torch.isinf(model_delta_vector).any():
            self.logger.warning(f"Client {self.client_idx}: NaN/Inf detected in delta after potential clipping. Replacing with zeros.")
            model_delta_vector = torch.zeros_like(model_delta_vector)
        
        return model_delta_vector.cpu() # Return delta on CPU

    def perform_local_training_and_get_raw_delta(self):
        """Main method called by server: trains locally and returns the raw model delta vector."""
        avg_acc, avg_loss = self.train() 
        model_delta = self.get_model_delta_vector()
        
        num_samples = self.num_train # Assuming self.num_train is set
        self.logger.debug(f"Client {self.client_idx} sending raw delta. Samples: {num_samples}, Acc: {avg_acc:.2f}%, Loss: {avg_loss:.4f}")
        
        if model_delta is None: # Handle case where delta calculation failed
             self.logger.error(f"Client {self.client_idx}: Model delta is None. Sending 0 samples and None delta.")
             return None, avg_acc, avg_loss, 0 # Or handle as appropriate

        return model_delta, avg_acc, avg_loss, num_samples

    # get_update method can be kept if FedAvg-style state_dict updates are needed for other server types
    # For FedFope/FedGFT, the delta vector is primary.
    def get_update(self, global_model_state_dict):
        """
        Calculates the update vector (difference based on state_dicts).
        This is more aligned with how FedAvg client might calculate delta if needed for other purposes.
        FedFope/FedGFT uses get_model_delta_vector for its primary mechanism.
        """
        update = {}
        # self.latest_params should contain the state_dict after local training on CPU
        # Ensure self.model is available and contains the latest parameters
        if self.model is None:
            self.logger.error(f"Client {self.client_idx}: self.model not available for get_update.")
            return None

        local_params_state_dict = self.model.to('cpu').state_dict()

        for name, global_param_tensor in global_model_state_dict.items():
            if name in local_params_state_dict:
                local_param_tensor = local_params_state_dict[name] # Already on CPU
                delta = local_param_tensor - global_param_tensor.to("cpu")
                if torch.isnan(delta).any() or torch.isinf(delta).any():
                    self.logger.warning(f"Warning: NaN/Inf detected in delta for param {name} in client {self.client_idx} (get_update). Appending zeros.")
                    delta = torch.zeros_like(delta)
                update[name] = delta
            else:
                self.logger.warning(f"Warning: Parameter {name} not found in local model during get_update for client {self.client_idx}.")
        return update