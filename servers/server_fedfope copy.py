# In server_fedfope.py

import time
from copy import deepcopy
import torch
from servers.server_base import Server # 假设 Server 基类存在
from clients.client_fedfope import ClientFedFope # 确保 ClientFedFope 已正确定义
import logging
import traceback # 用于打印更详细的错误信息

# --- Utility Functions (ideally in a utils.py or similar) ---
def get_model_params_vector(model, device='cpu'):
    """Flattens all model parameters that require gradients into a single 1D tensor."""
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param.data.view(-1).to(device))
    return torch.cat(params) if params else torch.empty(0, device=device)

def set_model_params_from_vector(model, params_vector, device, logger_instance):
    """Sets model parameters from a 1D tensor."""
    offset = 0
    if params_vector.numel() == 0 and sum(p.numel() for p in model.parameters() if p.requires_grad) == 0:
        logger_instance.debug("Both model and params_vector have 0 grad elements. Skipping set_model_params.")
        return
    if params_vector.numel() == 0 and sum(p.numel() for p in model.parameters() if p.requires_grad) > 0:
        logger_instance.error("Parameter vector is empty but model expects grad parameters. Cannot set model params.")
        return


    for param in model.parameters():
        if not param.requires_grad:
            continue
        param_len = param.numel()
        if offset + param_len > params_vector.numel():
            logger_instance.error(f"Error: Param vector too short. Needed {offset + param_len}, got {params_vector.numel()}. Param shape: {param.shape}")
            logger_instance.error(f"Current offset: {offset}, current param_len: {param_len}")
            # To debug which parameter causes this:
            # for name, p_debug in model.named_parameters():
            #     if p_debug.requires_grad:
            #         logger_instance.error(f"Debug: Param {name}, Len: {p_debug.numel()}")
            return 
        try:
            # Ensure slice is on the target device for view_as, then copy to param's original device
            param_data_slice = params_vector[offset:offset+param_len].to(param.device).view_as(param.data)
            param.data.copy_(param_data_slice)
        except Exception as e:
            logger_instance.error(f"Error setting param: {e}. Param shape: {param.data.shape}, Slice shape attempting: {params_vector[offset:offset+param_len].shape}")
            logger_instance.error(traceback.format_exc())
            raise
        offset += param_len
    
    if offset != params_vector.numel():
        logger_instance.warning(f"Warning: Parameter vector length {params_vector.numel()} "
                                f"does not match accumulated model parameter length {offset}.")

def inverse_fourier_transform_aggregate(fourier_vector_aggregate):
    """Performs inverse FFT and returns the real part."""
    if not isinstance(fourier_vector_aggregate, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if not torch.is_complex(fourier_vector_aggregate):
         # This can happen if all coefficients become zero after L1 thresholding
         return fourier_vector_aggregate.real # Ensure it's real
    return torch.fft.ifft(fourier_vector_aggregate).real
# --- End Utility Functions ---

class ServerFedFope(Server):
    def __init__(self, args):
        super().__init__(args) 

        logger_name = getattr(self, 'id', 'ServerFed_DefaultID')
        self.logger = logging.getLogger(str(logger_name))
        if not self.logger.hasHandlers(): # Avoid duplicate handlers
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
        self.logger.setLevel(getattr(logging, "INFO", logging.INFO))
        
        self.clients = []
        self.num_clients = getattr(args, 'num_clients', 0)
        for client_idx in range(self.num_clients):
            c = ClientFedFope(args, client_idx) # Pass args to client
            self.clients.append(c)
        
        # FedFope specific parameters from args
        self.N_train_equivalent = getattr(args, 'fourier_N_train_equivalent', 1024)
        self.floor_frequency_cutoff_enabled = getattr(args, 'floor_frequency_cutoff_enabled', True)
        self.fourier_trim_percentage = getattr(args, 'fourier_trim_percentage', 0.1) # For trimmed mean
        self.fourier_l1_lambda = getattr(args, 'fourier_l1_lambda', 0.0) # For L1 regularization, 0.0 means no L1.

        # Ensure self.model is initialized (typically by Server base class)
        if not hasattr(self, 'model') or self.model is None:
            self.logger.critical("CRITICAL: self.model is not initialized in ServerFedFope!")
            # This is a fatal error, an actual model instance is required.
            # For placeholder if base class is expected to do it:
            # from models.cnn import CIFARNet # Or your specific model
            # self.model = CIFARNet(num_classes=args.num_classes) # Example
            raise ValueError("Server model not initialized.")

        self.device = getattr(args, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device) # Move global model to server's device

        # Calculate model_param_len_ based on parameters that require gradients
        self.model_param_len_ = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.model_param_len_ == 0 and self.num_clients > 0 : # if there are clients, model should have params
             self.logger.warning("Warning: Global model has 0 trainable parameters. Fourier operations might be trivial or fail.")


        # Other necessary attributes from args or base class
        self.global_rounds = getattr(args, 'global_rounds', 200)
        self.eval_gap = getattr(args, 'eval_gap', 1)
        # Ensure 'clients_per_round' is available from args or base class
        self.clients_per_round = getattr(args, 'clients_per_round', self.num_clients if self.num_clients > 0 else 1)

    def sample_active_clients(self):
        """Samples clients for the current round."""
        if self.num_clients == 0:
            self.active_client_indices = []
            self.active_clients = []
            return

        num_to_sample = min(self.clients_per_round, self.num_clients)
        self.active_client_indices = torch.randperm(self.num_clients)[:num_to_sample].tolist()
        self.active_clients = [self.clients[i] for i in self.active_client_indices]
        self.logger.debug(f"Sampled clients for round: {self.active_client_indices}")

    def send_models_to_active_clients(self):
        """Sends the current global model (instance) to the active clients."""
        if not self.active_clients:
            self.logger.warning("No active clients to send models to.")
            return
        if self.model is None:
            self.logger.error("Global model (self.model) is not available to send.")
            return
        
        self.model.eval() # Ensure model is in eval mode before sending
        for client_obj in self.active_clients:
            client_obj.set_model(self.model) # Client's set_model handles deepcopy and device transfer


    def train(self):
        self.logger.info(f"Starting FedFope training with {self.num_clients} clients, {self.clients_per_round} per round.")
        self.logger.info(f"Hyperparameters: N_train_equivalent={self.N_train_equivalent}, floor_cutoff={self.floor_frequency_cutoff_enabled}, "
                         f"trim_percentage={self.fourier_trim_percentage}, l1_lambda={self.fourier_l1_lambda}")

        for r in range(1, self.global_rounds + 1):
            round_start_time = time.time()
            
            self.sample_active_clients()
            if not self.active_clients:
                self.logger.warning(f"Round {r}: No clients sampled, skipping round.")
                # Add to round_times to maintain length consistency if needed by plotting etc.
                self.round_times.append(time.time() - round_start_time) 
                continue
            
            self.send_models_to_active_clients()
            
            # Train clients and collect their Fourier updates
            train_loss, train_acc = self.train_round() # This now returns avg loss/acc
            
            elapsed_round_time = time.time() - round_start_time
            self.round_times.append(elapsed_round_time)

            log_message = f"Round [{r}/{self.global_rounds}]\t Avg Client Train Loss: {train_loss:.4f}\t Avg Client Train Acc: {train_acc:.2f}\t Round Time: {elapsed_round_time:.2f}s"

            if r % self.eval_gap == 0 or r == self.global_rounds:
                test_acc, test_loss, test_acc_std = self.evaluate_global_model()
                log_message += f"\t Global Test Loss: {test_loss:.4f}\t Global Test Acc: {test_acc:.2f} (std:{test_acc_std:.2f})"
                
                # Optional: Personalized evaluation
                ptest_acc, ptest_loss, ptest_acc_std = self.evaluate_personalized()
                log_message += f"\t Personalized Test Loss: {ptest_loss:.4f}\t Personalized Test Acc: {ptest_acc:.2f} (std:{ptest_acc_std:.2f})"
            
            self.logger.info(log_message)
        self.logger.info("FedFope training finished.")

    def train_clients_and_collect_fourier_updates(self):
        collected_fourier_updates = []
        client_sample_weights = []
        total_loss_weighted = 0.0
        total_acc_weighted = 0.0
        total_samples_in_round = 0

        if not self.active_clients: # Changed from active_client_indices as active_clients is more direct
            self.logger.error("active_clients not set or empty. Cannot train clients.")
            return [], [], 0.0, 0.0
        
        client_results = []
        for client_obj in self.active_clients: # Iterate over client objects
            try:
                # Client trains, calculates delta, performs FFT, and returns
                f_update, acc, loss, n_samples = client_obj.perform_local_training_and_get_fourier_update()
                client_results.append({'f_update': f_update, 'acc': acc, 'loss': loss, 'n_samples': n_samples, 'client_idx': client_obj.client_idx})
            except Exception as e:
                self.logger.error(f"Error during training or update generation for client {client_obj.client_idx}: {e}")
                self.logger.error(traceback.format_exc())
                client_results.append({'f_update': None, 'acc': 0, 'loss': float('inf'), 'n_samples': 0, 'client_idx': client_obj.client_idx})


        for res in client_results:
            if res['f_update'] is not None and res['n_samples'] > 0:
                # Ensure f_update is on the server's device for aggregation
                collected_fourier_updates.append(res['f_update'].to(self.device))
                client_sample_weights.append(res['n_samples'])
                total_loss_weighted += res['loss'] * res['n_samples']
                total_acc_weighted += res['acc'] * res['n_samples']
                total_samples_in_round += res['n_samples']
                self.logger.debug(f"Client {res['client_idx']} trained. Acc: {res['acc']:.2f}, Loss: {res['loss']:.4f}, Samples: {res['n_samples']}")
            else:
                self.logger.warning(f"Client {res['client_idx']} returned no valid Fourier update or had 0 samples.")
        
        avg_train_loss = (total_loss_weighted / total_samples_in_round) if total_samples_in_round > 0 else 0.0
        avg_train_acc = (total_acc_weighted / total_samples_in_round) if total_samples_in_round > 0 else 0.0
        
        return collected_fourier_updates, client_sample_weights, avg_train_loss, avg_train_acc

    def aggregate_fourier_updates(self, fourier_updates, client_weights):
        if not fourier_updates:
            self.logger.warning("No Fourier updates to aggregate.")
            return None

        if self.model_param_len_ == 0 and fourier_updates: # Attempt to infer if not set (e.g. model had no grad params initially)
             self.logger.warning("model_param_len_ is 0. Attempting to infer from the first update.")
             self.model_param_len_ = fourier_updates[0].shape[0]
             if self.model_param_len_ == 0:
                 self.logger.error("Could not determine model_param_len_ from updates (length is 0). Aggregation failed.")
                 return None
        elif self.model_param_len_ == 0: # No updates and model_param_len is 0
            self.logger.error("model_param_len_ is 0 and no updates to infer from. Aggregation failed.")
            return None

        frequencies = torch.fft.fftfreq(self.model_param_len_, device=self.device)
        processed_updates = []
        valid_update_indices = [] # Keep track of original indices of updates that are valid

        for i, f_update in enumerate(fourier_updates):
            if f_update.shape[0] != self.model_param_len_:
                self.logger.error(f"Mismatched Fourier update length ({f_update.shape[0]} vs {self.model_param_len_}) for update index {i}. Skipping.")
                continue
            
            f_update_processed = f_update.clone() 

            if self.floor_frequency_cutoff_enabled and self.N_train_equivalent > 0:
                floor_freq_thresh = 1.0 / self.N_train_equivalent
                indices_to_zero_out = (torch.abs(frequencies) < floor_freq_thresh) & (frequencies != 0) # Exclude DC
                f_update_processed[indices_to_zero_out] = 0.0 + 0.0j # Zero out complex number
                if torch.sum(indices_to_zero_out).item() > 0:
                     self.logger.debug(f"Applied floor frequency cutoff to update index {i}. Threshold: {floor_freq_thresh:.4e}. Zeroed {torch.sum(indices_to_zero_out).item()} components.")
            
            processed_updates.append(f_update_processed)
            valid_update_indices.append(i)

        if not processed_updates:
            self.logger.warning("No updates left after processing (e.g. all mismatched or floor cutoff zeroed all).")
            return None

        adjusted_client_weights_list = [client_weights[i] for i in valid_update_indices]
        if not adjusted_client_weights_list: # Should not happen if processed_updates is not empty
             self.logger.error("No client weights for processed updates. This is unexpected.")
             return None

        adjusted_client_weights = torch.tensor(adjusted_client_weights_list, device=self.device, dtype=torch.float32)
        total_weight = torch.sum(adjusted_client_weights)

        if total_weight == 0: 
            self.logger.warning("Total client weight for processed updates is 0. Using equal weights for processed updates.")
            adjusted_client_weights = torch.ones(len(processed_updates), device=self.device, dtype=torch.float32)
            total_weight = torch.sum(adjusted_client_weights)
        
        if total_weight == 0: # Still zero (e.g., no processed_updates even after trying equal weights)
             self.logger.error("Cannot aggregate with zero total weight and no processed updates remaining.")
             return None

        # Stack processed updates: (num_valid_clients, model_param_len_)
        stacked_updates = torch.stack(processed_updates, dim=0)
        aggregated_f_update = torch.zeros(self.model_param_len_, dtype=torch.complex64, device=self.device)

        # --- Robust Aggregation: Trimmed Mean or Weighted Average ---
        if self.fourier_trim_percentage > 0 and len(processed_updates) > 2: # Need at least 3 for trimming
            num_valid_clients = stacked_updates.shape[0]
            # Number of elements to trim from each end for each frequency component
            k_trim = int(num_valid_clients * self.fourier_trim_percentage)

            if k_trim > 0 and (num_valid_clients - 2 * k_trim > 0):
                self.logger.info(f"Applying trimmed mean aggregation: trim_percentage={self.fourier_trim_percentage}, k_trim={k_trim} from each end.")
                # Sort along the client dimension for each frequency component's real and imaginary parts
                sorted_updates_real = torch.sort(stacked_updates.real, dim=0).values
                sorted_updates_imag = torch.sort(stacked_updates.imag, dim=0).values
                
                trimmed_updates_real = sorted_updates_real[k_trim : num_valid_clients - k_trim, :]
                trimmed_updates_imag = sorted_updates_imag[k_trim : num_valid_clients - k_trim, :]
                
                # For simplicity, using unweighted mean of trimmed values.
                # For weighted trimmed mean, select corresponding weights, average, and normalize.
                aggregated_f_update_real = torch.mean(trimmed_updates_real, dim=0)
                aggregated_f_update_imag = torch.mean(trimmed_updates_imag, dim=0)
                aggregated_f_update = aggregated_f_update_real + 1j * aggregated_f_update_imag
            else:
                self.logger.info(f"Not enough clients ({num_valid_clients}) to trim {k_trim} elements or k_trim is 0. Falling back to weighted average.")
                normalized_weights = adjusted_client_weights / total_weight
                aggregated_f_update = torch.sum(stacked_updates * normalized_weights.view(-1, 1), dim=0)
        else: # Fall back to simple weighted average if trim_percentage is 0 or too few clients
            normalized_weights = adjusted_client_weights / total_weight
            aggregated_f_update = torch.sum(stacked_updates * normalized_weights.view(-1, 1), dim=0)
        
        # --- L1 Regularization (Soft Thresholding) on Aggregated Fourier Update ---
        if self.fourier_l1_lambda > 0:
            self.logger.info(f"Applying L1 soft-thresholding with lambda={self.fourier_l1_lambda}.")
            threshold_val = self.fourier_l1_lambda
            
            sign_real = torch.sign(aggregated_f_update.real)
            sign_imag = torch.sign(aggregated_f_update.imag)
            
            abs_real_minus_thresh = torch.clamp(torch.abs(aggregated_f_update.real) - threshold_val, min=0.0)
            abs_imag_minus_thresh = torch.clamp(torch.abs(aggregated_f_update.imag) - threshold_val, min=0.0)
            
            aggregated_f_update = (sign_real * abs_real_minus_thresh) + 1j * (sign_imag * abs_imag_minus_thresh)
            
            num_zeroed = torch.sum((aggregated_f_update.real == 0.0) & (aggregated_f_update.imag == 0.0)).item()
            self.logger.debug(f"L1 soft-thresholding zeroed out {num_zeroed}/{self.model_param_len_} Fourier coefficients.")
            if num_zeroed == self.model_param_len_:
                self.logger.warning("L1 soft-thresholding zeroed out ALL Fourier coefficients. Global update will be zero.")


        aggregated_delta_vector = inverse_fourier_transform_aggregate(aggregated_f_update)
        return aggregated_delta_vector.to(self.device)


    def apply_aggregated_update_to_global_model(self, aggregated_delta_vector):
        if aggregated_delta_vector is None:
            self.logger.warning("No aggregated delta to apply. Global model remains unchanged.")
            return
        if self.model is None:
            self.logger.error("Global model (self.model) is not available to apply update.")
            return

        self.model = self.model.to(self.device) # Ensure model is on the correct device
        current_global_params = get_model_params_vector(self.model, device=self.device)
        
        if current_global_params.numel() == 0 and aggregated_delta_vector.numel() > 0:
            self.logger.error("Global model has no grad parameters, but received a non-empty delta.")
            return
        if current_global_params.numel() == 0 and aggregated_delta_vector.numel() == 0:
             self.logger.debug("Global model has no grad parameters, and delta is empty. No update applied.")
             return

        if current_global_params.numel() != aggregated_delta_vector.numel():
            self.logger.error(f"Mismatch between global model param length ({current_global_params.numel()}) "
                              f"and aggregated delta vector length ({aggregated_delta_vector.numel()}). Cannot apply update.")
            return

        updated_global_params = current_global_params + aggregated_delta_vector.to(current_global_params.device)
        set_model_params_from_vector(self.model, updated_global_params, device=self.device, logger_instance=self.logger)
        self.logger.info("Global model updated with aggregated Fourier delta.")


    def train_round(self):
        f_updates, client_weights, round_avg_loss, round_avg_acc = self.train_clients_and_collect_fourier_updates()

        if not f_updates:
            self.logger.warning("No valid Fourier updates received from clients this round. Skipping model update.")
            return round_avg_loss, round_avg_acc # Return collected stats even if no update

        aggregated_delta = self.aggregate_fourier_updates(f_updates, client_weights)

        if aggregated_delta is not None:
            self.apply_aggregated_update_to_global_model(aggregated_delta)
        else:
            self.logger.warning("Aggregated delta is None. Global model not updated this round.")
        
        return round_avg_loss, round_avg_acc


    def evaluate_global_model(self): # Renamed for clarity from 'evaluate' if 'evaluate' is used for personalized
        self.logger.info("Evaluating global model on all clients...")
        if self.model is None:
            self.logger.error("Global model is not available for evaluation.")
            return 0.0, 0.0, 0.0
        
        all_client_original_states = []
        for c in self.clients: # Store original states if clients manage their own fine-tuned models
            all_client_original_states.append(deepcopy(c.model.state_dict()))
            c.set_model(self.model) # Client gets a deepcopy of current global model for evaluation

        # Call the generic evaluation method (which iterates through self.clients)
        # Assuming self.evaluate() is defined in server_base.py or here
        # and evaluates c.model for each client 'c' on their respective test sets.
        # For now, let's define a simple version if not inherited from server_base.py
        
        if not self.clients:
            self.logger.warning("No clients to evaluate.")
            return 0.0, 0.0, 0.0 

        total_samples = sum(getattr(c, 'num_test', 0) for c in self.clients)
        if total_samples == 0:
            self.logger.warning("Total test samples across all clients is 0 for global evaluation.")
            return 0.0, 0.0, 0.0 

        weighted_loss = 0.0
        weighted_acc = 0.0
        accs_list = [] 
        
        for i, c in enumerate(self.clients):
            # Client's model (c.model) is now the global model
            acc_val, loss_val = c.evaluate() # Client's evaluate method
            
            loss_item = loss_val.item() if isinstance(loss_val, torch.Tensor) else float(loss_val)
            acc_item = acc_val.item() if isinstance(acc_val, torch.Tensor) else float(acc_val)

            accs_list.append(torch.tensor(acc_item, device='cpu'))
            
            client_num_test = getattr(c, 'num_test', 0)
            if client_num_test > 0:
                weighted_loss += (client_num_test / total_samples) * loss_item
                weighted_acc += (client_num_test / total_samples) * acc_item
            
            # Restore client's original model state
            # c.model.load_state_dict(all_client_original_states[i])
            # c.model.to(c.device)


        std_dev = torch.std(torch.stack(accs_list)).item() if accs_list else 0.0
        self.logger.info(f"Global model evaluation - Weighted Acc: {weighted_acc:.2f}%, Weighted Loss: {weighted_loss:.4f}, Acc StdDev: {std_dev:.2f}")
        return weighted_acc, weighted_loss, std_dev


    def evaluate_personalized(self):
        self.logger.info("Evaluating personalized models on all clients...")
        if not self.clients:
            self.logger.warning("No clients for personalized evaluation.")
            return 0.0, 0.0, 0.0

        if self.model is None: # Global model as base for personalization
            self.logger.error("Global model is not available as base for personalization.")
            return 0.0, 0.0, 0.0
            
        for c in self.clients:
            c.set_model(self.model) # Client gets a copy of the current global model
            self.logger.debug(f"Personalizing model for client {c.client_idx} by local training...")
            c.train() # Client trains locally to personalize the model
            # Now c.model is the client's personalized model
        
        # Now evaluate these personalized models (c.model for each client)
        # This reuses the logic from evaluate_global_model, but now c.model is personalized.
        # For clarity, we can call a generic evaluation method.
        # Assuming self.evaluate() in server_base.py evaluates c.model.
        
        # Re-implementing eval logic here for clarity if not cleanly inherited
        total_samples = sum(getattr(c, 'num_test', 0) for c in self.clients)
        if total_samples == 0:
            self.logger.warning("Total test samples across all clients is 0 for personalized evaluation.")
            return 0.0, 0.0, 0.0

        weighted_loss = 0.0
        weighted_acc = 0.0
        accs_list = []
        
        for c in self.clients:
            # c.model is now the personalized model
            acc_val, loss_val = c.evaluate()
            
            loss_item = loss_val.item() if isinstance(loss_val, torch.Tensor) else float(loss_val)
            acc_item = acc_val.item() if isinstance(acc_val, torch.Tensor) else float(acc_val)
            accs_list.append(torch.tensor(acc_item, device='cpu'))
            
            client_num_test = getattr(c, 'num_test', 0)
            if client_num_test > 0:
                weighted_loss += (client_num_test / total_samples) * loss_item
                weighted_acc += (client_num_test / total_samples) * acc_item
                
        std_dev = torch.std(torch.stack(accs_list)).item() if accs_list else 0.0
        self.logger.info(f"Personalized model evaluation - Weighted Acc: {weighted_acc:.2f}%, Weighted Loss: {weighted_loss:.4f}, Acc StdDev: {std_dev:.2f}")
        return weighted_acc, weighted_loss, std_dev