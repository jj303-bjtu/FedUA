# In server_fedfope.py

import time
from copy import deepcopy
import torch
import numpy as np # For GFT placeholder with random orthogonal matrix if needed
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
    expected_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if params_vector.numel() == 0 and expected_total_params == 0:
        logger_instance.debug("Both model and params_vector have 0 grad elements. Skipping set_model_params.")
        return
    if params_vector.numel() == 0 and expected_total_params > 0:
        logger_instance.error("Parameter vector is empty but model expects grad parameters. Cannot set model params.")
        return
    if params_vector.numel() != expected_total_params:
        logger_instance.error(f"Parameter vector length {params_vector.numel()} does not match "
                              f"model's expected parameter length {expected_total_params}. Cannot set params.")
        return

    for param in model.parameters():
        if not param.requires_grad:
            continue
        param_len = param.numel()
        # This check is largely covered by the initial total length check
        # if offset + param_len > params_vector.numel(): 
        #     # ... error logging ...
        #     return 
        try:
            param_data_slice = params_vector[offset:offset+param_len].to(param.device).view_as(param.data)
            param.data.copy_(param_data_slice)
        except Exception as e:
            logger_instance.error(f"Error setting param: {e}. Param shape: {param.data.shape}, Slice shape attempting: {params_vector[offset:offset+param_len].shape}")
            logger_instance.error(traceback.format_exc())
            raise
        offset += param_len
    
    # This warning should ideally not be triggered if the initial length check is correct
    # if offset != params_vector.numel():
    #     logger_instance.warning(f"Warning: Parameter vector length {params_vector.numel()} "
    #                             f"does not match accumulated model parameter length {offset} after setting.")

# Inverse FFT remains for the FFT path
def inverse_fourier_transform_aggregate(fourier_vector_aggregate):
    """Performs inverse FFT and returns the real part."""
    if not isinstance(fourier_vector_aggregate, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    if not torch.is_complex(fourier_vector_aggregate):
         return fourier_vector_aggregate.real 
    return torch.fft.ifft(fourier_vector_aggregate).real
# --- End Utility Functions ---

class ServerFedFope(Server):
    def __init__(self, args):
        super().__init__(args) 

        logger_name = getattr(self, 'id', 'ServerFedFope_DefaultID')
        self.logger = logging.getLogger(str(logger_name))
        if not self.logger.hasHandlers(): 
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
        self.logger.setLevel(getattr(logging, 'INFO', logging.INFO))
        
        self.clients = []
        self.num_clients = getattr(args, 'num_clients', 0)
        for client_idx in range(self.num_clients):
            c = ClientFedFope(args, client_idx) 
            self.clients.append(c)
        
        self.N_train_equivalent = getattr(args, 'fourier_N_train_equivalent', 1024)
        self.floor_frequency_cutoff_enabled = getattr(args, 'floor_frequency_cutoff_enabled', True)
        self.fourier_trim_percentage = getattr(args, 'fourier_trim_percentage', 0.1) 
        self.fourier_l1_lambda = getattr(args, 'fourier_l1_lambda', 0.0)

        self.use_gft = getattr(args, 'use_gft', False) 
        self.max_params_for_gft_placeholder = getattr(args, 'max_params_for_gft_placeholder', 2048)
        self.graph_eigenvectors_V = None 
        self.graph_eigenvectors_Vt = None


        if not hasattr(self, 'model') or self.model is None:
            self.logger.critical("CRITICAL: self.model is not initialized in ServerFedFope!")
            raise ValueError("Server model not initialized.")

        self.device = getattr(args, 'device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device) 

        self.model_param_len_ = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.model_param_len_ == 0 and self.num_clients > 0 :
             self.logger.warning("Warning: Global model has 0 trainable parameters. Transform operations might be trivial or fail.")
        
        if self.use_gft: # Attempt to initialize GFT placeholders if use_gft is true
            self._initialize_graph_representation_placeholder()
        # If _initialize_graph_representation_placeholder sets self.use_gft to False (e.g. due to size), 
        # then GFT path will be disabled in subsequent logic.

        self.global_rounds = getattr(args, 'global_rounds', 200)
        self.eval_gap = getattr(args, 'eval_gap', 20)
        self.clients_per_round = getattr(args, 'clients_per_round', self.num_clients if self.num_clients > 0 else 1)
        if not hasattr(self, 'round_times'):
            self.round_times = []


    def _initialize_graph_representation_placeholder(self):
        self.logger.info("Attempting to initialize GFT graph representation (Placeholder).")
        if not self.use_gft: # Explicitly check if GFT is still enabled
            self.logger.info("GFT: use_gft is False. Skipping placeholder initialization.")
            return

        if self.model_param_len_ == 0:
            self.logger.warning("GFT: Model has no parameters, cannot initialize graph representation.")
            self.use_gft = False # Disable GFT if no params
            return

        if self.model_param_len_ > self.max_params_for_gft_placeholder:
            self.logger.warning(f"GFT: Model parameter length ({self.model_param_len_}) exceeds "
                                f"max_params_for_gft_placeholder ({self.max_params_for_gft_placeholder}). "
                                "GFT placeholder will not be used. Falling back to 1D FFT if applicable.")
            self.use_gft = False 
            return

        self.logger.warning("GFT: This is a PLACEHOLDER graph representation. "
                            "Actual GFT requires graph construction from model architecture (e.g., defining a "
                            "graph Laplacian L based on parameter connectivity/relations) and its eigendecomposition (V, D = eig(L)). "
                            "The following uses a random orthogonal matrix for V to allow code execution, IT IS NOT A REAL GFT.")
        try:
            # Placeholder V: A random orthogonal matrix. V.T = V_inv for orthogonal.
            # GFT(x) = V.T @ x
            # IGFT(c) = V @ c
            # The problematic line was here:
            temp_v = torch.randn(self.model_param_len_, self.model_param_len_, device=self.device, dtype=torch.float32) 
            
            if self.model_param_len_ > 0: # Ensure Q is not empty
                q, _ = torch.linalg.qr(temp_v) # Q is orthogonal (Q.T Q = I)
                self.graph_eigenvectors_V = q      # This is our placeholder V
                self.graph_eigenvectors_Vt = q.T   # This is our placeholder V.T (V_transpose)
                self.logger.info(f"GFT Placeholder: Initialized V (shape {self.graph_eigenvectors_V.shape}) "
                                 f"and V.T (shape {self.graph_eigenvectors_Vt.shape}) with random orthogonal matrices.")
            else: # Should have been caught by self.model_param_len_ == 0 earlier
                self.logger.warning("GFT: model_param_len_ is 0, cannot create QR decomposition.")
                self.use_gft = False


        except Exception as e:
            self.logger.error(f"Error creating placeholder GFT eigenvectors: {e}")
            self.logger.error(traceback.format_exc())
            self.use_gft = False # Disable GFT if placeholder creation fails
            self.graph_eigenvectors_V = None
            self.graph_eigenvectors_Vt = None

    def graph_fourier_transform(self, data_vector):
        """Performs placeholder GFT: V.T @ data_vector."""
        if self.graph_eigenvectors_Vt is None or not self.use_gft:
            # This condition should ideally be caught before calling, or handle fallback explicitly
            self.logger.warning("GFT V.T not initialized or GFT not enabled. Using FFT as fallback for transform.")
            return torch.fft.fft(data_vector.cfloat()) # Ensure complex float for FFT
        
        data_vector_prepared = data_vector.to(device=self.graph_eigenvectors_Vt.device, dtype=self.graph_eigenvectors_Vt.dtype)
        # V.T is real, data_vector is real, so GFT coeffs are real
        gft_coeffs = torch.matmul(self.graph_eigenvectors_Vt, data_vector_prepared)
        return gft_coeffs # These are real coefficients

    def inverse_graph_fourier_transform(self, gft_coeffs):
        """Performs placeholder IGFT: V @ gft_coefficients."""
        if self.graph_eigenvectors_V is None or not self.use_gft:
            self.logger.warning("GFT V not initialized or GFT not enabled. Using IFFT as fallback for inverse transform.")
            # Ensure gft_coeffs are complex for IFFT if they came from FFT path
            return torch.fft.ifft(gft_coeffs.cfloat() if not torch.is_complex(gft_coeffs) else gft_coeffs).real

        # Assuming gft_coeffs are real as per the output of our placeholder graph_fourier_transform
        coeffs_prepared = gft_coeffs.to(device=self.graph_eigenvectors_V.device, dtype=self.graph_eigenvectors_V.dtype)
        
        transformed_signal = torch.matmul(self.graph_eigenvectors_V, coeffs_prepared)
        # Output is real since V is real and gft_coeffs are assumed real
        return transformed_signal # .real is implicitly handled if dtypes are real


    def sample_active_clients(self):
        """Samples clients for the current round."""
        if self.num_clients == 0:
            self.active_client_indices = []
            self.active_clients = []
            return

        num_to_sample = min(self.clients_per_round, self.num_clients)
        # Ensure torch.randperm is used correctly
        self.active_client_indices = torch.randperm(self.num_clients, device='cpu')[:num_to_sample].tolist() # Generate on CPU
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
        
        self.model.eval() 
        for client_obj in self.active_clients:
            client_obj.set_model(self.model)


    def train(self):
        mode = "GFT (Placeholder)" if self.use_gft and self.graph_eigenvectors_V is not None else "FFT"
        self.logger.info(f"Starting FedFope training in {mode} mode with {self.num_clients} clients, {self.clients_per_round} per round.")
        if mode == "FFT":
            self.logger.info(f"FFT Hyperparameters: N_train_equivalent={self.N_train_equivalent}, floor_cutoff={self.floor_frequency_cutoff_enabled}, "
                             f"trim_percentage={self.fourier_trim_percentage}, l1_lambda={self.fourier_l1_lambda}")
        else: # GFT mode
             self.logger.info(f"GFT (Placeholder) Hyperparameters: trim_percentage={self.fourier_trim_percentage}, l1_lambda={self.fourier_l1_lambda}")
             self.logger.warning("GFT floor frequency cutoff based on Laplacian eigenvalues is not implemented in this placeholder version.")


        for r in range(1, self.global_rounds + 1):
            round_start_time = time.time()
            
            self.sample_active_clients()
            if not self.active_clients:
                self.logger.warning(f"Round {r}: No clients sampled, skipping round.")
                if hasattr(self, 'round_times') and isinstance(self.round_times, list):
                    self.round_times.append(time.time() - round_start_time) 
                continue
            
            self.send_models_to_active_clients()
            
            train_loss, train_acc = self.train_round() 
            
            elapsed_round_time = time.time() - round_start_time
            if hasattr(self, 'round_times') and isinstance(self.round_times, list):
                 self.round_times.append(elapsed_round_time)

            log_message = f"Round [{r}/{self.global_rounds}]\t Mode: {mode}\t Avg Client Train Loss: {train_loss:.4f}\t Avg Client Train Acc: {train_acc:.2f}\t Round Time: {elapsed_round_time:.2f}s"

            if r % self.eval_gap == 0 or r == self.global_rounds:
                # Ensure evaluate_global_model and evaluate_personalized are defined and compatible
                test_acc, test_loss, test_acc_std = self.evaluate_global_model()
                log_message += f"\t Global Test Loss: {test_loss:.4f}\t Global Test Acc: {test_acc:.2f} (std:{test_acc_std:.2f})"
                
                ptest_acc, ptest_loss, ptest_acc_std = self.evaluate_personalized()
                log_message += f"\t Personalized Test Loss: {ptest_loss:.4f}\t Personalized Test Acc: {ptest_acc:.2f} (std:{ptest_acc_std:.2f})"
            
            self.logger.info(log_message)
        self.logger.info(f"FedFope training ({mode} mode) finished.")

    def train_clients_and_collect_transformed_updates(self):
        collected_transformed_updates = []
        client_sample_weights = []
        total_loss_weighted = 0.0
        total_acc_weighted = 0.0
        total_samples_in_round = 0

        if not self.active_clients:
            self.logger.error("active_clients not set or empty. Cannot train clients.")
            return [], [], 0.0, 0.0
        
        client_results = []
        for client_obj in self.active_clients: 
            try:
                delta_vector, acc, loss, n_samples = client_obj.perform_local_training_and_get_raw_delta()
                
                transformed_update = None
                if delta_vector is not None:
                    delta_vector_dev = delta_vector.to(self.device) # Ensure delta_vector is on server device
                    if self.use_gft and self.graph_eigenvectors_V is not None and self.graph_eigenvectors_Vt is not None: # GFT Path
                        # GFT takes real vector, produces real coeffs with our placeholder
                        transformed_update = self.graph_fourier_transform(delta_vector_dev.float()) # Use float for GFT input
                    else: # Fallback or default FFT Path
                        # FFT takes real/complex, produces complex. Ensure input is float/complex.
                        transformed_update = torch.fft.fft(delta_vector_dev.cfloat()) # Use cfloat for FFT
                
                client_results.append({'transformed_update': transformed_update, 
                                       'acc': acc, 'loss': loss, 'n_samples': n_samples, 
                                       'client_idx': client_obj.client_idx})
            except Exception as e:
                self.logger.error(f"Error during training or update transformation for client {client_obj.client_idx}: {e}")
                self.logger.error(traceback.format_exc())
                client_results.append({'transformed_update': None, 'acc': 0, 'loss': float('inf'), 
                                       'n_samples': 0, 'client_idx': client_obj.client_idx})

        for res in client_results:
            if res['transformed_update'] is not None and res['n_samples'] > 0:
                # Transformed updates should already be on self.device from GFT/FFT step
                collected_transformed_updates.append(res['transformed_update']) 
                client_sample_weights.append(res['n_samples'])
                total_loss_weighted += res['loss'] * res['n_samples']
                total_acc_weighted += res['acc'] * res['n_samples']
                total_samples_in_round += res['n_samples']
                self.logger.debug(f"Client {res['client_idx']} contributed. Acc: {res['acc']:.2f}, Loss: {res['loss']:.4f}, Samples: {res['n_samples']}")
            else:
                self.logger.warning(f"Client {res['client_idx']} returned no valid transformed update or had 0 samples.")
        
        avg_train_loss = (total_loss_weighted / total_samples_in_round) if total_samples_in_round > 0 else 0.0
        avg_train_acc = (total_acc_weighted / total_samples_in_round) if total_samples_in_round > 0 else 0.0
        
        return collected_transformed_updates, client_sample_weights, avg_train_loss, avg_train_acc

    # aggregate_transformed_updates needs to handle real GFT coeffs vs complex FFT coeffs
    def aggregate_transformed_updates(self, transformed_updates, client_weights_list):
        if not transformed_updates:
            self.logger.warning("No transformed updates to aggregate.")
            return None

        if self.model_param_len_ == 0:
            self.logger.error("model_param_len_ is 0. Cannot aggregate updates.")
            return None
        
        client_weights = torch.tensor(client_weights_list, device=self.device, dtype=torch.float32)
        total_weight = torch.sum(client_weights)
        if total_weight == 0:
            self.logger.warning("Total client weight is 0. Using equal weights for aggregation.")
            client_weights = torch.ones(len(transformed_updates), device=self.device, dtype=torch.float32)
            total_weight = torch.sum(client_weights)
            if total_weight == 0:
                 self.logger.error("Cannot aggregate with zero total weight even after attempting equal weights.")
                 return None
        
        normalized_client_weights = (client_weights / total_weight).type(transformed_updates[0].dtype) # Match dtype for multiplication
        
        # stacked_updates might be complex (FFT) or real (GFT placeholder)
        stacked_updates = torch.stack(transformed_updates, dim=0) 

        # --- GFT Path ---
        if self.use_gft and self.graph_eigenvectors_V is not None and self.graph_eigenvectors_Vt is not None:
            self.logger.debug("Aggregating in GFT (Placeholder) domain.")
            # GFT coefficients are real with the current placeholder (V.T @ real_delta)
            # Ensure operations are on real numbers.
            
            current_coeffs_are_real = not torch.is_complex(stacked_updates)
            if not current_coeffs_are_real: # Should not happen if GFT path is consistent
                self.logger.warning("GFT Path: Expected real GFT coefficients but received complex. Check GFT pipeline. Proceeding with .real part.")
                stacked_updates = stacked_updates.real
            
            aggregated_coeffs = None
            num_clients_participating = stacked_updates.shape[0]

            if self.fourier_trim_percentage > 0 and num_clients_participating > 2:
                k_trim = int(num_clients_participating * self.fourier_trim_percentage)
                if k_trim > 0 and (num_clients_participating - 2 * k_trim > 0):
                    self.logger.info(f"GFT Path: Applying trimmed mean: trim_percentage={self.fourier_trim_percentage}, k_trim={k_trim}.")
                    sorted_gft_coeffs = torch.sort(stacked_updates, dim=0).values 
                    trimmed_gft_coeffs = sorted_gft_coeffs[k_trim : num_clients_participating - k_trim, :]
                    aggregated_coeffs = torch.mean(trimmed_gft_coeffs, dim=0)
                else:
                    self.logger.info(f"GFT Path: Not enough clients ({num_clients_participating}) to trim {k_trim} or k_trim is 0. Using weighted average.")
                    aggregated_coeffs = torch.sum(stacked_updates * normalized_client_weights.view(-1, 1), dim=0)
            else: 
                aggregated_coeffs = torch.sum(stacked_updates * normalized_client_weights.view(-1, 1), dim=0)

            if self.fourier_l1_lambda > 0 and aggregated_coeffs is not None:
                self.logger.info(f"GFT Path: Applying L1 soft-thresholding with lambda={self.fourier_l1_lambda}.")
                sign_coeffs = torch.sign(aggregated_coeffs)
                abs_coeffs_minus_thresh = torch.clamp(torch.abs(aggregated_coeffs) - self.fourier_l1_lambda, min=0.0)
                aggregated_coeffs = sign_coeffs * abs_coeffs_minus_thresh
                num_zeroed = torch.sum(aggregated_coeffs == 0.0).item()
                self.logger.debug(f"GFT L1 soft-thresholding zeroed out {num_zeroed}/{self.model_param_len_} GFT coefficients.")

            if aggregated_coeffs is None: return None
            aggregated_delta_vector = self.inverse_graph_fourier_transform(aggregated_coeffs) # Takes real, returns real
            return aggregated_delta_vector.to(self.device)

        # --- FFT Path (Original FedFope) ---
        else:
            self.logger.debug("Aggregating in FFT domain.")
            # FFT coefficients are complex
            if not torch.is_complex(stacked_updates): # Should not happen if FFT path is consistent
                 self.logger.warning("FFT Path: Expected complex FFT coefficients but received real. Check FFT pipeline. Casting to complex.")
                 stacked_updates = stacked_updates.cfloat()

            frequencies = torch.fft.fftfreq(self.model_param_len_, device=self.device)
            processed_updates_fft = []

            for i, f_update in enumerate(transformed_updates): 
                if f_update.shape[0] != self.model_param_len_:
                    self.logger.error(f"Mismatched FFT update length for update index {i}. Skipping.")
                    continue
                
                f_update_processed = f_update.clone() 
                if self.floor_frequency_cutoff_enabled and self.N_train_equivalent > 0:
                    floor_freq_thresh = 1.0 / self.N_train_equivalent
                    indices_to_zero_out = (torch.abs(frequencies) < floor_freq_thresh) & (frequencies != 0)
                    f_update_processed[indices_to_zero_out] = 0.0 + 0.0j 
                processed_updates_fft.append(f_update_processed)
            
            if not processed_updates_fft:
                self.logger.warning("FFT Path: No updates left after processing. Cannot aggregate.")
                return None
            
            stacked_fft_updates = torch.stack(processed_updates_fft, dim=0)
            aggregated_fft_coeffs = None
            num_clients_participating_fft = stacked_fft_updates.shape[0]

            if self.fourier_trim_percentage > 0 and num_clients_participating_fft > 2:
                k_trim_fft = int(num_clients_participating_fft * self.fourier_trim_percentage)
                if k_trim_fft > 0 and (num_clients_participating_fft - 2 * k_trim_fft > 0):
                    self.logger.info(f"FFT Path: Applying trimmed mean: trim_percentage={self.fourier_trim_percentage}, k_trim={k_trim_fft}.")
                    sorted_updates_real = torch.sort(stacked_fft_updates.real, dim=0).values
                    sorted_updates_imag = torch.sort(stacked_fft_updates.imag, dim=0).values
                    trimmed_updates_real = sorted_updates_real[k_trim_fft : num_clients_participating_fft - k_trim_fft, :]
                    trimmed_updates_imag = sorted_updates_imag[k_trim_fft : num_clients_participating_fft - k_trim_fft, :]
                    aggregated_fft_coeffs_real = torch.mean(trimmed_updates_real, dim=0)
                    aggregated_fft_coeffs_imag = torch.mean(trimmed_updates_imag, dim=0)
                    aggregated_fft_coeffs = aggregated_fft_coeffs_real + 1j * aggregated_fft_coeffs_imag
                else:
                    self.logger.info(f"FFT Path: Not enough clients ({num_clients_participating_fft}) to trim {k_trim_fft} or k_trim is 0. Using weighted average.")
                    aggregated_fft_coeffs = torch.sum(stacked_fft_updates * normalized_client_weights.view(-1, 1), dim=0)
            else: 
                aggregated_fft_coeffs = torch.sum(stacked_fft_updates * normalized_client_weights.view(-1, 1), dim=0)

            if self.fourier_l1_lambda > 0 and aggregated_fft_coeffs is not None:
                self.logger.info(f"FFT Path: Applying L1 soft-thresholding with lambda={self.fourier_l1_lambda}.")
                threshold_val = self.fourier_l1_lambda
                sign_real = torch.sign(aggregated_fft_coeffs.real)
                sign_imag = torch.sign(aggregated_fft_coeffs.imag)
                abs_real_minus_thresh = torch.clamp(torch.abs(aggregated_fft_coeffs.real) - threshold_val, min=0.0)
                abs_imag_minus_thresh = torch.clamp(torch.abs(aggregated_fft_coeffs.imag) - threshold_val, min=0.0)
                aggregated_fft_coeffs = (sign_real * abs_real_minus_thresh) + 1j * (sign_imag * abs_imag_minus_thresh)
                num_zeroed = torch.sum((aggregated_fft_coeffs.real == 0.0) & (aggregated_fft_coeffs.imag == 0.0)).item()
                self.logger.debug(f"FFT L1 soft-thresholding zeroed out {num_zeroed}/{self.model_param_len_} FFT coefficients.")

            if aggregated_fft_coeffs is None: return None
            aggregated_delta_vector = inverse_fourier_transform_aggregate(aggregated_fft_coeffs) # Takes complex, returns real
            return aggregated_delta_vector.to(self.device)


    def apply_aggregated_update_to_global_model(self, aggregated_delta_vector):
        if aggregated_delta_vector is None:
            self.logger.warning("No aggregated delta to apply. Global model remains unchanged.")
            return
        if self.model is None:
            self.logger.error("Global model (self.model) is not available to apply update.")
            return

        self.model = self.model.to(self.device) 
        current_global_params = get_model_params_vector(self.model, device=self.device)
        
        if current_global_params.numel() == 0 and aggregated_delta_vector.numel() > 0 :
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
        self.logger.info("Global model updated with aggregated delta.")


    def train_round(self):
        transformed_updates, client_weights, round_avg_loss, round_avg_acc = self.train_clients_and_collect_transformed_updates()

        if not transformed_updates:
            self.logger.warning("No valid transformed updates received from clients this round. Skipping model update.")
            return round_avg_loss, round_avg_acc 

        aggregated_delta = self.aggregate_transformed_updates(transformed_updates, client_weights)

        if aggregated_delta is not None:
            self.apply_aggregated_update_to_global_model(aggregated_delta)
        else:
            self.logger.warning("Aggregated delta is None. Global model not updated this round.")
        
        return round_avg_loss, round_avg_acc


    def evaluate_global_model(self): 
        self.logger.info("Evaluating global model on all clients...")
        if self.model is None:
            self.logger.error("Global model is not available for evaluation.")
            return 0.0, 0.0, 0.0
        
        if not self.clients:
            self.logger.warning("No clients to evaluate.")
            return 0.0, 0.0, 0.0 

        # Store original client models if they are stateful beyond just holding the global model temporarily
        # For evaluating the *current global model*, clients just need a copy.
        # original_client_models_states = [deepcopy(c.model.state_dict()) for c in self.clients]

        for c in self.clients: 
            # MODIFIED LINE: Use set_model instead of set_model_for_evaluation
            c.set_model(self.model) 

        # Call the generic evaluation method from base class or implement here
        # This part assumes clients have an `evaluate` method and `num_test` attribute
        
        total_samples = sum(getattr(c, 'num_test', 0) for c in self.clients if hasattr(c, 'evaluate'))
        if total_samples == 0:
            self.logger.warning("Total test samples across all clients is 0 for global evaluation.")
            # It's better to return 0 for accuracy if no samples, rather than potentially a NaN or error from division by zero later.
            return 0.0, 0.0, 0.0 # Acc, Loss, Std_Dev

        weighted_loss = 0.0
        weighted_acc = 0.0 # This will be sum of (acc * weight_fraction), so effectively already weighted
        accs_list = [] 
        
        for i, c in enumerate(self.clients):
            if not hasattr(c, 'evaluate'):
                self.logger.warning(f"Client {c.client_idx} does not have an evaluate method. Skipping.")
                continue

            # Client's c.model is now the global model copy
            acc_val, loss_val = c.evaluate() # Client's evaluate method
            
            # Ensure loss_val and acc_val are scalars
            loss_item = loss_val.item() if isinstance(loss_val, torch.Tensor) else float(loss_val)
            acc_item = acc_val.item() if isinstance(acc_val, torch.Tensor) else float(acc_val) # acc_val from client.evaluate() is likely already %

            accs_list.append(torch.tensor(acc_item, device='cpu')) # Collect accs for std dev
            
            client_num_test = getattr(c, 'num_test', 0)
            if client_num_test > 0: 
                # Correct weighting calculation:
                # weighted_loss += (client_num_test / total_samples) * loss_item # This is if loss_item is an average
                # weighted_acc += (client_num_test / total_samples) * acc_item   # This is if acc_item is an average
                # A more common way is to sum up total correct predictions and total loss * samples, then divide by total_samples
                # Assuming client c.evaluate() returns average loss and accuracy for that client's test set.
                weighted_loss += loss_item * client_num_test
                weighted_acc += acc_item * client_num_test # acc_item is already in percentage (e.g. 90.0 for 90%)
            
            # Restore client's original model state if necessary
            # c.model.load_state_dict(original_client_models_states[i])
            # c.model.to(c.device) # Move back to client's device

        final_weighted_loss = weighted_loss / total_samples if total_samples > 0 else 0.0
        final_weighted_acc = weighted_acc / total_samples if total_samples > 0 else 0.0 # This is now correctly weighted average accuracy

        std_dev = torch.std(torch.stack(accs_list)).item() if accs_list else 0.0
        self.logger.info(f"Global model evaluation - Weighted Acc: {final_weighted_acc:.2f}%, Weighted Loss: {final_weighted_loss:.4f}, Acc StdDev: {std_dev:.2f}")
        return final_weighted_acc, final_weighted_loss, std_dev


    def evaluate_personalized(self):
        self.logger.info("Evaluating personalized models on all clients...")
        if not self.clients:
            self.logger.warning("No clients for personalized evaluation.")
            return 0.0, 0.0, 0.0

        if self.model is None: 
            self.logger.error("Global model is not available as base for personalization.")
            return 0.0, 0.0, 0.0
            
        for c in self.clients:
            if not hasattr(c, 'train') or not hasattr(c, 'set_model'): 
                 self.logger.warning(f"Client {c.client_idx} cannot be personalized (missing train/set_model). Skipping.")
                 continue
            # MODIFIED LINE: Use set_model here as well.
            # When c.train() is called next, initial_global_params_vector will be based on this current global model.
            c.set_model(self.model) 
            self.logger.debug(f"Personalizing model for client {c.client_idx} by local training...")
            c.train() # Client trains locally, self.model becomes personalized. initial_global_params_vector in client is reset.
        
        total_samples = sum(getattr(c, 'num_test', 0) for c in self.clients if hasattr(c, 'evaluate'))
        if total_samples == 0:
            self.logger.warning("Total test samples across all clients is 0 for personalized evaluation.")
            return 0.0, 0.0, 0.0 # Acc, Loss, Std_Dev

        weighted_loss = 0.0
        weighted_acc = 0.0 # This will be sum of (acc * num_samples)
        accs_list = []
        
        for c in self.clients:
            if not hasattr(c, 'evaluate'):
                 self.logger.warning(f"Client {c.client_idx} does not have an evaluate method. Skipping personalized eval for this client.")
                 continue
            
            # c.model is now the personalized model
            acc_val, loss_val = c.evaluate() 
            
            loss_item = loss_val.item() if isinstance(loss_val, torch.Tensor) else float(loss_val)
            acc_item = acc_val.item() if isinstance(acc_val, torch.Tensor) else float(acc_val) # acc_val from client.evaluate() is likely already %
            accs_list.append(torch.tensor(acc_item, device='cpu'))
            
            client_num_test = getattr(c, 'num_test', 0)
            if client_num_test > 0:
                weighted_loss += loss_item * client_num_test
                weighted_acc += acc_item * client_num_test
                
        final_weighted_loss = weighted_loss / total_samples if total_samples > 0 else 0.0
        final_weighted_acc = weighted_acc / total_samples if total_samples > 0 else 0.0

        std_dev = torch.std(torch.stack(accs_list)).item() if accs_list else 0.0
        self.logger.info(f"Personalized model evaluation - Weighted Acc: {final_weighted_acc:.2f}%, Weighted Loss: {final_weighted_loss:.4f}, Acc StdDev: {std_dev:.2f}")
        return final_weighted_acc, final_weighted_loss, std_dev

    # Add set_model_for_evaluation to client or ensure base Client has it
    # For now, assuming clients `set_model` is used for evaluation purposes too.
    # If clients need to preserve their own distinct models during global eval, that needs more sophisticated state management.