import torch
import torch.nn.functional as F
from ripser import ripser
import numpy as np
from clients.client_base import Client # Assuming ClientBase has self.logger and methods like load_train_data, load_test_data, set_model, loss
from utils.util import AverageMeter # Assuming this utility is available
import traceback
import logging

SAFE_REPLACEMENT_VALUE=1e20

class ClientFedAvg(Client):
    def __init__(self, args, client_idx, is_corrupted=False):
        super().__init__(args, client_idx, is_corrupted=is_corrupted)
        self.tda_subsample_size = getattr(args, 'tda_subsample_size', 500)
        self.tda_dim = getattr(args, 'tda_dim', 1) # Max dimension for homology (H0, H1)
        self.tda_on_activations = getattr(args, 'tda_on_activations', True)

        # Ensure logger is available
        if not hasattr(self, 'logger') or self.logger is None:
            logger_name_prefix = getattr(args, 'client_logger_name_prefix', 'ClientFedTop') # Match 'ClientFedFope' if that's your prefix
            self.logger = logging.getLogger(f"{logger_name_prefix}_{client_idx}")
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        self.logger.debug(f"Client {self.client_idx} initialized. TDA on activations: {self.tda_on_activations}, TDA subsample size: {self.tda_subsample_size}, TDA dim: {self.tda_dim}")

    def train(self):
        trainloader = self.load_train_data() # Assumes this method from ClientBase loads with default batch_size and drop_last settings for training
        if trainloader is None:
            self.logger.error(f"Client {self.client_idx}: Train data loader is None. Skipping training.")
            return 0.0, float('inf')

        try:
            if hasattr(trainloader, 'dataset') and len(trainloader.dataset) == 0:
                self.logger.warning(f"Client {self.client_idx}: Training data loader's dataset is empty. Skipping training.")
                return 0.0, float('inf')
            if hasattr(trainloader, '__len__') and len(trainloader) == 0 :
                 self.logger.warning(f"Client {self.client_idx}: Training data loader yields 0 batches. Skipping training.")
                 return 0.0, float('inf')
        except Exception as e:
            self.logger.warning(f"Client {self.client_idx}: Could not determine if trainloader is empty ({e}). Proceeding with caution.")

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
        self.model = self.model.to(self.device)
        self.model.train()
        losses = AverageMeter()
        accs = AverageMeter()

        actual_batches_processed = 0
        for e in range(self.local_epochs):
            try:
                for i, (x, y) in enumerate(trainloader):
                    actual_batches_processed +=1
                    x = x.to(self.device)
                    y = y.to(self.device)

                    output = self.model(x)
                    loss_val = self.loss(output, y) 

                    acc_val = (output.argmax(1) == y).float().mean() * 100.0
                    accs.update(acc_val.item(), x.size(0))
                    losses.update(loss_val.item(), x.size(0))

                    optimizer.zero_grad()
                    loss_val.backward()
                    optimizer.step()
            except StopIteration:
                self.logger.warning(f"Client {self.client_idx}: Trainloader became empty during epoch {e}.")
                break 
            if actual_batches_processed == 0 and e == 0 : 
                self.logger.warning(f"Client {self.client_idx}: No batches were processed in the first training epoch. Trainloader might be effectively empty.")
                break

        self.model = self.model.to("cpu")
        if actual_batches_processed == 0:
            self.logger.warning(f"Client {self.client_idx}: No data batches were processed during the entire training phase.")
            return 0.0, float('inf') 

        return accs.avg, losses.avg

    def get_embeddings_for_tda_with_model(self, model_to_use, data_sample_x):
        model_to_use = model_to_use.to(self.device)
        model_to_use.eval()

        if not self.tda_on_activations:
            self.logger.error("get_embeddings_for_tda_with_model called when tda_on_activations is False. This function is for activations.")
            return np.array([])

        if data_sample_x is None:
            self.logger.warning(f"Client {self.client_idx}: Data sample is None for TDA on activations.")
            return np.array([])

        data_sample_x = data_sample_x.to(self.device)
        embeddings_np = np.array([])

        try:
            with torch.no_grad():
                activation_output = None
                hook_handle = None

                def hook_fn(module, input, output):
                    nonlocal activation_output
                    activation_output = output.detach().clone()

                target_layer = None
                children = list(model_to_use.children())
                if hasattr(model_to_use, 'features') and isinstance(model_to_use.features, torch.nn.Module):
                    target_layer = model_to_use.features
                elif len(children) > 1 :
                    if isinstance(children[-1], torch.nn.Linear) : 
                        if len(children) > 1:
                             target_layer = children[-2] 
                        else: 
                             target_layer = children[-1]
                    else: 
                         target_layer = children[-1]
                elif len(children) == 1: 
                    target_layer = children[0]

                if target_layer:
                    hook_handle = target_layer.register_forward_hook(hook_fn)
                else: 
                    self.logger.debug(f"Client {self.client_idx}: No specific target layer found for TDA hook, will use full model output.")

                _ = model_to_use(data_sample_x)

                if hook_handle:
                    hook_handle.remove()

                if activation_output is None: 
                    self.logger.debug(f"Client {self.client_idx}: TDA hook for activations not effective or no target layer, using direct model output for activations.")
                    activation_output = model_to_use(data_sample_x) # Re-run if hook didn't capture

                current_embeddings = activation_output.cpu().numpy().reshape(activation_output.size(0), -1)

                if current_embeddings.shape[0] > self.tda_subsample_size:
                    indices = np.random.choice(current_embeddings.shape[0], self.tda_subsample_size, replace=False)
                    embeddings_np = current_embeddings[indices]
                else:
                    embeddings_np = current_embeddings
        except Exception as e:
            self.logger.error(f"Client {self.client_idx}: Error extracting activations: {e}\n{traceback.format_exc()}")
            return np.array([])

        return embeddings_np

    def get_embeddings_for_tda(self, data_loader_for_activations):
        self.model.eval()

        if self.tda_on_activations:
            if data_loader_for_activations is None:
                self.logger.error(f"Client {self.client_idx}: Data loader is None but TDA on activations is requested.")
                return np.array([])

            underlying_dataset = None
            if hasattr(data_loader_for_activations, 'dataset'):
                underlying_dataset = data_loader_for_activations.dataset
                dataset_size = len(underlying_dataset)
                loader_batch_size = data_loader_for_activations.batch_size if hasattr(data_loader_for_activations, 'batch_size') else 'N/A'
                self.logger.debug(f"Client {self.client_idx}: TDA DataLoader - Dataset size: {dataset_size}, Batch size: {loader_batch_size}, Drop_last: {getattr(data_loader_for_activations, 'drop_last', 'N/A')}")
                if dataset_size == 0:
                    self.logger.warning(f"Client {self.client_idx}: DataLoader's underlying dataset is EMPTY for TDA. Cannot get activations.")
                    return np.array([])

            if hasattr(data_loader_for_activations, '__len__'):
                num_batches = len(data_loader_for_activations)
                self.logger.debug(f"Client {self.client_idx}: TDA DataLoader reports it will yield {num_batches} batch(es).")
                if num_batches == 0: # This check is important
                     # This warning might still appear if drop_last=False was ignored by base class or dataset truly became empty
                     if underlying_dataset and len(underlying_dataset) > 0:
                         self.logger.warning(f"Client {self.client_idx}: TDA DataLoader reports 0 batches despite non-empty dataset (size {len(underlying_dataset)}) and attempt to use drop_last=False. Check base load_train_data behavior.")
                     else:
                         self.logger.warning(f"Client {self.client_idx}: TDA DataLoader reports 0 batches (dataset might be empty or issue with loader config).")
                     return np.array([])
            try:
                loader_iter = iter(data_loader_for_activations)
                x_sample, _ = next(loader_iter)
                self.logger.debug(f"Client {self.client_idx}: Successfully obtained a batch of size {x_sample.shape[0]} for TDA on activations.")
                return self.get_embeddings_for_tda_with_model(self.model, x_sample)
            except StopIteration:
                self.logger.warning(f"Client {self.client_idx}: Data loader was empty upon iteration (StopIteration) for TDA on activations. This can happen if drop_last=False was ineffective or dataset is truly empty.")
                return np.array([])
            except Exception as e:
                self.logger.error(f"Client {self.client_idx}: Error iterating data_loader_for_activations for TDA: {e}\n{traceback.format_exc()}")
                return np.array([])
        else: # TDA on weights
            try:
                weight_list = []
                for param in self.model.parameters():
                    weight_list.append(param.data.view(-1).cpu().numpy())
                if not weight_list:
                    self.logger.warning(f"Client {self.client_idx}: No weights found in model for TDA.")
                    return np.array([])

                all_weights = np.concatenate(weight_list)
                if all_weights.size == 0:
                    self.logger.warning(f"Client {self.client_idx}: Concatenated weights are empty for TDA.")
                    return np.array([])

                if all_weights.shape[0] > self.tda_subsample_size:
                    indices = np.random.choice(all_weights.shape[0], self.tda_subsample_size, replace=False)
                    embeddings_np = all_weights[indices]
                else:
                    embeddings_np = all_weights
                return embeddings_np.reshape(-1, 1) # Ensure 2D for ripser
            except Exception as e:
                self.logger.error(f"Client {self.client_idx}: Error extracting weights for TDA: {e}\n{traceback.format_exc()}")
                return np.array([])

    def compute_persistence_diagrams(self, embeddings_np):
        if embeddings_np is None or embeddings_np.size == 0:
            self.logger.warning(f"Client {self.client_idx}: Embeddings for PD computation are None or empty.")
            return {'diagrams': [np.array([]) for _ in range(self.tda_dim + 1)], 'error': True, 'client_idx': self.client_idx}

        try:
            if embeddings_np.ndim == 1: # Ensure embeddings are 2D for ripser
                embeddings_np = embeddings_np.reshape(-1, 1)

            if np.isnan(embeddings_np).any() or np.isinf(embeddings_np).any():
                self.logger.warning(f"Client {self.client_idx}: NaN/Inf found in embeddings before ripser. Replacing with safe finite values.")
                embeddings_np = np.nan_to_num(embeddings_np, nan=0.0, posinf=SAFE_REPLACEMENT_VALUE, neginf=-SAFE_REPLACEMENT_VALUE) 

            embeddings_np = embeddings_np.astype(np.float64) # Ripser prefers float64

            if embeddings_np.shape[0] < 2 : # Ripser needs at least 2 points for meaningful topology beyond H0 counts
                self.logger.warning(f"Client {self.client_idx}: Not enough data points ({embeddings_np.shape[0]}) for ripser after preprocessing. Returning empty diagrams.")
                return {'diagrams': [np.array([]) for _ in range(self.tda_dim + 1)], 'error': True, 'client_idx': self.client_idx}

            pd_result = ripser(embeddings_np, maxdim=self.tda_dim, thresh=float('inf')) 
            raw_diagrams = pd_result['dgms']

            processed_diagrams = []
            for i in range(self.tda_dim + 1): # Ensure we cover H0 up to tda_dim
                if i < len(raw_diagrams):
                    diag = raw_diagrams[i]
                    diag = np.nan_to_num(diag, nan=0.0, posinf=SAFE_REPLACEMENT_VALUE if diag.size >0 else 0.0 , neginf=0.0) 
                    processed_diagrams.append(diag if len(diag) > 0 else np.array([]).reshape(0,2))
                else:
                    processed_diagrams.append(np.array([]).reshape(0,2)) 

            return {'diagrams': processed_diagrams, 'client_idx': self.client_idx, 'error': False}
        except Exception as e:
            self.logger.error(f"Client {self.client_idx}: Error in Ripser computation: {e}\n{traceback.format_exc()}")
            return {'diagrams': [np.array([]) for _ in range(self.tda_dim + 1)], 'error': True, 'client_idx': self.client_idx}

    def get_current_pd(self):
        try:
            data_loader = None
            if self.tda_on_activations:
                self.logger.debug(f"Client {self.client_idx} (get_current_pd): Attempting to load train data for TDA with batch_size={self.tda_subsample_size} and drop_last=False.")
                data_loader = self.load_train_data(batch_size=self.tda_subsample_size, drop_last=False)
                if data_loader is None:
                    self.logger.warning(f"Client {self.client_idx} (get_current_pd): load_train_data returned None (with drop_last=False attempt) for TDA on activations. Returning no PD.")
                    return None
                self.logger.debug(f"Client {self.client_idx} (get_current_pd): Successfully called load_train_data for TDA.")

            embeddings = self.get_embeddings_for_tda(data_loader)

            if embeddings is None or embeddings.size == 0:
                self.logger.warning(f"Client {self.client_idx} (get_current_pd): Embeddings for TDA are None or empty. This might be due to an empty data loader or issues in activation extraction.")
                return None

            pd_result_dict = self.compute_persistence_diagrams(embeddings)
            if pd_result_dict['error']:
                self.logger.warning(f"Client {self.client_idx} (get_current_pd): Computation of PD resulted in an error.")
                return None
            return pd_result_dict
        except Exception as e:
            self.logger.error(f"Client {self.client_idx}: Error in get_current_pd: {e}\n{traceback.format_exc()}")
            return None

    def get_pd_features(self):
        data_loader = None
        if self.tda_on_activations:
            self.logger.debug(f"Client {self.client_idx} (get_pd_features): Attempting to load train data for TDA with batch_size={self.tda_subsample_size} and drop_last=False.")
            data_loader = self.load_train_data(batch_size=self.tda_subsample_size, drop_last=False)
            if data_loader is None:
                 self.logger.warning(f"Client {self.client_idx} (get_pd_features): load_train_data returned None (with drop_last=False attempt) for TDA on activations.")
                 return {'diagrams': [np.array([]) for _ in range(self.tda_dim + 1)], 'error': True, 'client_idx': self.client_idx}
            self.logger.debug(f"Client {self.client_idx} (get_pd_features): Successfully called load_train_data for TDA.")

        embeddings = self.get_embeddings_for_tda(data_loader)
        if embeddings is None or embeddings.size == 0:
             self.logger.warning(f"Client {self.client_idx} (get_pd_features): Embeddings for TDA are None or empty for post-training PD. This might be due to an empty data loader or issues in activation extraction.")
             return {'diagrams': [np.array([]) for _ in range(self.tda_dim + 1)], 'error': True, 'client_idx': self.client_idx}
        return self.compute_persistence_diagrams(embeddings)

    def get_update(self, global_model_state_dict):
        local_model_state_dict = self.model.state_dict()
        update_params = []
        try:
            processed_param_names = set()
            for name, global_param_tensor in global_model_state_dict.items():
                processed_param_names.add(name)
                if name in local_model_state_dict:
                    local_param_tensor = local_model_state_dict[name].to("cpu")
                    delta = local_param_tensor - global_param_tensor.to("cpu").detach().clone()
                    if torch.isnan(delta).any() or torch.isinf(delta).any():
                        self.logger.warning(f"Client {self.client_idx}: NaN/Inf detected in delta for param {name}. Using zeros for this param.")
                        delta = torch.zeros_like(delta)
                    update_params.append(delta.view(-1))
                else:
                    self.logger.warning(f"Client {self.client_idx}: Parameter {name} (shape {global_param_tensor.shape}) from global model not found in local model during get_update. Appending zeros for this param update.")
                    update_params.append(torch.zeros(global_param_tensor.numel(), dtype=torch.float32, device="cpu"))
            
            for name in local_model_state_dict:
                if name not in processed_param_names:
                    self.logger.warning(f"Client {self.client_idx}: Local parameter {name} not found in global model state_dict during get_update. It will not be part of the update.")

            if not update_params:
                 self.logger.error(f"Client {self.client_idx}: No parameters processed for update. Model structures might mismatch or local model is empty.")
                 return torch.tensor([], dtype=torch.float32, device="cpu")

            return torch.cat(update_params)
        except Exception as e:
            self.logger.error(f"Client {self.client_idx}: Error calculating update: {e}\n{traceback.format_exc()}", exc_info=True)
            return torch.tensor([], dtype=torch.float32, device="cpu")