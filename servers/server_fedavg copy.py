import time
import torch
from copy import deepcopy
import numpy as np
from persim import wasserstein, bottleneck
import random
from servers.server_base import Server # Assuming ServerBase has self.logger, model, clients etc.
from clients.client_fedavg import ClientFedAvg # Use the modified client class
import traceback
import logging

class ServerFedAvg(Server):
    def __init__(self, args):
        super().__init__(args)
        if not hasattr(self, 'logger') or self.logger is None:
            logger_name = getattr(args, 'server_logger_name', 'ServerFedTop') 
            self.logger = logging.getLogger(logger_name)
            if not self.logger.hasHandlers():
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

        self.clients = [ClientFedAvg(args, client_idx) for client_idx in range(self.num_clients)]
        if not self.clients:
             self.logger.critical("No clients were initialized. Server cannot operate.")
             self.model = None 
        else:
            if self.model is None: 
                 self.model = deepcopy(self.clients[0].model) 
                 self.logger.info("Server model initialized from client 0's model structure.")

        # self.num_active_clients is assumed to be initialized by ServerBase to the 
        # CONFIGURED INTEGER number of clients to be active per round (e.g., from args.frac_clients or args.num_clients_per_round).
        # This is crucial for the fix.

        self.aggregation_strategy = getattr(args, 'aggregation_strategy', 'fedtop_select_diverse') # fedtop_select_diverse, fedtopavg, fedavg
        self.topo_weight_lambda = getattr(args, 'topo_weight_lambda', 0.1)
        self.pd_distance_metric = getattr(args, 'pd_distance_metric', 'wasserstein')
        self.tda_homology_dim_to_use = getattr(args, 'tda_homology_dim_to_use', 1) 
        self.tda_on_activations = getattr(args, 'tda_on_activations', True) 
        self.tda_subsample_size = getattr(args, 'tda_subsample_size', 500) 

        self.candidate_pool_factor = getattr(args, 'candidate_pool_factor', 1.5)
        self.diversity_selection_w1 = getattr(args, 'diversity_selection_w1', 1.0)
        self.diversity_selection_w2 = getattr(args, 'diversity_selection_w2', 0.5)
        
        self.fedtopavg_similarity_scale = getattr(args, 'fedtopavg_similarity_scale', 1.0)

        self.reference_pd_for_fedtopavg = None 
        self.num_active_clients = len(self.active_clients)

        self.logger.info(f"Server initialized with aggregation strategy: {self.aggregation_strategy}")
        if self.aggregation_strategy == 'fedtopavg':
            self.logger.info(f"FedTopAvg params: lambda={self.topo_weight_lambda}, pd_metric={self.pd_distance_metric}, H_dim_server_use={self.tda_homology_dim_to_use}, similarity_scale={self.fedtopavg_similarity_scale}")
        elif self.aggregation_strategy == 'fedtop_select_diverse':
            self.logger.info(f"FedTopSelectDiverse params: pool_factor={self.candidate_pool_factor}, w1_global_sim={self.diversity_selection_w1}, w2_diversity={self.diversity_selection_w2}")

    def pd_distance(self, pd1_list, pd2_list, homology_dim_to_use):
        if pd1_list is None or pd2_list is None:
            self.logger.debug("PD list is None for distance calculation, returning max distance.")
            return float('inf')

        total_dist = 0.0
        num_dims_compared = 0
        max_dims_to_compare = homology_dim_to_use + 1 

        for dim_idx in range(max_dims_to_compare):
            dgm1 = pd1_list[dim_idx] if dim_idx < len(pd1_list) else np.array([]).reshape(0,2)
            dgm2 = pd2_list[dim_idx] if dim_idx < len(pd2_list) else np.array([]).reshape(0,2)

            if not isinstance(dgm1, np.ndarray) or dgm1.ndim != 2 or (dgm1.size > 0 and dgm1.shape[1] != 2):
                 self.logger.warning(f"PD distance: dgm1 for dim {dim_idx} is malformed ({type(dgm1)}, shape {dgm1.shape if isinstance(dgm1, np.ndarray) else 'N/A'}). Using empty diagram.")
                 dgm1 = np.array([]).reshape(0,2)
            if not isinstance(dgm2, np.ndarray) or dgm2.ndim != 2 or (dgm2.size > 0 and dgm2.shape[1] != 2):
                 self.logger.warning(f"PD distance: dgm2 for dim {dim_idx} is malformed ({type(dgm2)}, shape {dgm2.shape if isinstance(dgm2, np.ndarray) else 'N/A'}). Using empty diagram.")
                 dgm2 = np.array([]).reshape(0,2)
            
            dgm1 = dgm1.astype(np.float64) if dgm1.size > 0 else dgm1
            dgm2 = dgm2.astype(np.float64) if dgm2.size > 0 else dgm2

            dgm1 = dgm1[dgm1[:, 1] > dgm1[:, 0]] if dgm1.size > 0 else dgm1
            dgm1 = dgm1[~np.isnan(dgm1).any(axis=1)] if dgm1.size > 0 else dgm1
            dgm2 = dgm2[dgm2[:, 1] > dgm2[:, 0]] if dgm2.size > 0 else dgm2
            dgm2 = dgm2[~np.isnan(dgm2).any(axis=1)] if dgm2.size > 0 else dgm2

            dist_dim = 0.0
            if dgm1.size == 0 and dgm2.size == 0:
                dist_dim = 0.0
            else:
                try:
                    if self.pd_distance_metric == 'wasserstein':
                        dist_dim = wasserstein(dgm1, dgm2)
                    elif self.pd_distance_metric == 'bottleneck':
                        dist_dim = bottleneck(dgm1, dgm2)
                    else:
                        self.logger.warning(f"Unknown PD distance metric: {self.pd_distance_metric}. Defaulting to Wasserstein.")
                        dist_dim = wasserstein(dgm1, dgm2)
                except Exception as e:
                    self.logger.error(f"Error computing PD distance for dim {dim_idx}: {e}. Diag1 shape: {dgm1.shape}, Diag2 shape: {dgm2.shape}.\n{traceback.format_exc()}")
                    dist_dim = float('inf')

            if np.isnan(dist_dim) or np.isinf(dist_dim):
                dist_dim = float('inf') 

            total_dist += dist_dim
            num_dims_compared += 1
        
        if num_dims_compared == 0:
             return float('inf')
        return total_dist / num_dims_compared

    def get_global_model_pd_features(self):
        self.logger.debug("Computing PD for global model...")
        if not self.clients:
            self.logger.warning("No clients available to act as proxy for global PD computation.")
            return None
        if self.model is None:
            self.logger.warning("Global model is None. Cannot compute its PD features.")
            return None

        proxy_client = self.clients[0] 
        
        original_proxy_model_state = None
        if hasattr(proxy_client.model, 'state_dict'):
            original_proxy_model_state = deepcopy(proxy_client.model.state_dict())
        
        if isinstance(self.model, torch.nn.Module):
            proxy_client.model.load_state_dict(self.model.state_dict())
        else: 
            proxy_client.set_model(deepcopy(self.model))

        pd_info_diagrams = None
        try:
            pd_info_dict = proxy_client.get_current_pd() 
            if pd_info_dict and not pd_info_dict.get('error', True):
                pd_info_diagrams = pd_info_dict['diagrams']
                self.logger.debug(f"Successfully computed PD for global model via proxy client {proxy_client.client_idx}.")
            else:
                self.logger.warning(f"Failed to get PD features for global model via proxy client {proxy_client.client_idx}. PD_info: {pd_info_dict}")
        except Exception as e:
            self.logger.error(f"Exception computing global model PD via proxy client {proxy_client.client_idx}: {e}\n{traceback.format_exc()}")
        finally:
            if original_proxy_model_state is not None: 
                proxy_client.model.load_state_dict(original_proxy_model_state)

        return pd_info_diagrams

    # MODIFIED select_clients_topologically method
    def select_clients_topologically(self, candidate_clients_pool, num_to_select_target):
        self.logger.info(f"Starting topological client selection from {len(candidate_clients_pool)} candidates (target: {num_to_select_target})...")
        
        num_to_select = num_to_select_target # Use the passed configured target number

        # The problematic line `self.num_active_clients_per_round = len(self.active_clients)` has been removed.

        if not candidate_clients_pool: # Handle empty candidate pool
            self.logger.warning("Candidate pool for topological selection is empty. Cannot select clients.")
            return []

        if len(candidate_clients_pool) <= num_to_select:
            self.logger.info(f"Candidate pool size ({len(candidate_clients_pool)}) is less than or equal to target selection size ({num_to_select}). Using all candidates from pool.")
            return candidate_clients_pool # Return the whole pool if it's smaller or equal

        global_pd_features = self.get_global_model_pd_features()
        if global_pd_features is None:
            self.logger.warning("Global PD features not available for topological selection. Falling back to random selection from candidates.")
            return random.sample(candidate_clients_pool, min(num_to_select, len(candidate_clients_pool)))

        client_pds_info = []
        for client_obj in candidate_clients_pool:
            self.logger.debug(f"Requesting PD from candidate client {client_obj.client_idx} for selection.")
            pd_data = client_obj.get_current_pd() 
            if pd_data and not pd_data.get('error', True) and pd_data.get('diagrams') is not None:
                client_pds_info.append({'client': client_obj, 'pd_features': pd_data['diagrams']})
            else:
                self.logger.warning(f"Client {client_obj.client_idx} failed to provide valid PD for selection. PD_data: {pd_data}")

        if not client_pds_info :
            self.logger.warning(f"No clients provided valid PDs for topological selection. Falling back to random selection from candidates.")
            return random.sample(candidate_clients_pool, min(num_to_select, len(candidate_clients_pool)))
        
        # If fewer clients provided PDs than we want to select, we might take all of them and fill randomly,
        # or just take those who provided PDs if that's preferred over purely random ones.
        if len(client_pds_info) < num_to_select:
             self.logger.warning(f"Only {len(client_pds_info)} clients provided valid PDs, less than required {num_to_select}. Selecting all of them and attempting to fill randomly if needed.")
             selected_clients_from_pd = [info['client'] for info in client_pds_info]
             # Identify clients from the original candidate_pool that were not in client_pds_info (e.g., failed PD)
             # or are not already selected.
             current_selected_indices = {c.client_idx for c in selected_clients_from_pd}
             remaining_candidates_for_fill = [c for c in candidate_clients_pool if c.client_idx not in current_selected_indices]
             
             num_to_fill_randomly = num_to_select - len(selected_clients_from_pd)
             if num_to_fill_randomly > 0 and remaining_candidates_for_fill:
                 randomly_added = random.sample(remaining_candidates_for_fill, min(num_to_fill_randomly, len(remaining_candidates_for_fill)))
                 selected_clients_from_pd.extend(randomly_added)
                 self.logger.info(f"Added {len(randomly_added)} clients randomly to meet selection target.")
             self.logger.info(f"Selected {len(selected_clients_from_pd)} clients after PD issues and random fill: {[c.client_idx for c in selected_clients_from_pd]}")
             return selected_clients_from_pd


        selected_clients_final = []
        selected_pds_final = []

        for info in client_pds_info:
            info['dist_to_global'] = self.pd_distance(info['pd_features'], global_pd_features, self.tda_homology_dim_to_use)

        client_pds_info.sort(key=lambda x: x['dist_to_global']) 

        if client_pds_info: # Should always be true if we reached here due to `len(client_pds_info) >= num_to_select`
            first_client_info = client_pds_info.pop(0)
            selected_clients_final.append(first_client_info['client'])
            selected_pds_final.append(first_client_info['pd_features'])
            self.logger.debug(f"Topological selection: First client {first_client_info['client'].client_idx} (dist_global: {first_client_info['dist_to_global']:.4f})")
        else: 
            self.logger.error("Unexpected: No client PDs available after sorting for first pick. This path should not be reached if len(client_pds_info) >= num_to_select. Falling back to random.")
            return random.sample(candidate_clients_pool, min(num_to_select, len(candidate_clients_pool)))


        while len(selected_clients_final) < num_to_select and len(client_pds_info) > 0:
            best_candidate_info_for_round = None
            max_score_for_round = -float('inf')

            for candidate_info_item in client_pds_info:
                dist_to_global_term = -self.diversity_selection_w1 * candidate_info_item['dist_to_global']
                min_dist_to_already_selected_set = float('inf')
                if selected_pds_final: 
                    for sel_pd_item in selected_pds_final:
                        dist_val = self.pd_distance(candidate_info_item['pd_features'], sel_pd_item, self.tda_homology_dim_to_use)
                        if dist_val < min_dist_to_already_selected_set:
                            min_dist_to_already_selected_set = dist_val
                else: 
                    min_dist_to_already_selected_set = 0 
                diversity_term = self.diversity_selection_w2 * min_dist_to_already_selected_set
                current_score = dist_to_global_term + diversity_term
                candidate_info_item['score_debug'] = current_score

                if current_score > max_score_for_round:
                    max_score_for_round = current_score
                    best_candidate_info_for_round = candidate_info_item
            
            if self.logger.level == logging.DEBUG: 
                sorted_candidates_by_score = sorted(client_pds_info, key=lambda x: x.get('score_debug', -float('inf')), reverse=True)
                self.logger.debug(f"Topological selection iteration scoring (target {len(selected_clients_final)+1} of {num_to_select}):")
                for idx, cand_info in enumerate(sorted_candidates_by_score[:5]): 
                    self.logger.debug(f"  #{idx+1} Client {cand_info['client'].client_idx}: Score {cand_info.get('score_debug', -float('inf')):.4f} (d_glob: {cand_info['dist_to_global']:.4f})")


            if best_candidate_info_for_round:
                selected_clients_final.append(best_candidate_info_for_round['client'])
                selected_pds_final.append(best_candidate_info_for_round['pd_features'])
                client_pds_info.remove(best_candidate_info_for_round) 
                self.logger.debug(f"Topological selection: Added client {best_candidate_info_for_round['client'].client_idx} (score: {max_score_for_round:.4f}). Total selected: {len(selected_clients_final)}/{num_to_select}")
            else:
                self.logger.warning("No suitable best candidate found in topological selection iteration. Breaking.")
                break

        if len(selected_clients_final) < num_to_select:
            self.logger.warning(f"Only {len(selected_clients_final)} clients selected topologically. Required {num_to_select}. Attempting to fill randomly from remaining candidates.")
            current_selected_indices = {c.client_idx for c in selected_clients_final}
            remaining_pool_for_fill = [c for c in candidate_clients_pool if c.client_idx not in current_selected_indices]
            num_to_fill = num_to_select - len(selected_clients_final)

            if remaining_pool_for_fill:
                 num_can_fill = min(num_to_fill, len(remaining_pool_for_fill))
                 randomly_added_clients = random.sample(remaining_pool_for_fill, num_can_fill)
                 selected_clients_final.extend(randomly_added_clients)
                 self.logger.info(f"Filled {len(randomly_added_clients)} additional clients randomly.")
            else:
                 self.logger.warning(f"No remaining candidates in the original pool to fill the selection.")

        self.logger.info(f"Topologically selected {len(selected_clients_final)} clients: {[c.client_idx for c in selected_clients_final]}")
        return selected_clients_final

    def send_models(self, clients_to_send=None):
        target_clients = clients_to_send if clients_to_send is not None else self.active_clients
        if not target_clients:
            self.logger.debug("send_models called with no target clients.")
            return
        
        if self.model is None:
            self.logger.error("Global model is None, cannot send to clients.")
            return

        global_model_state = deepcopy(self.model.state_dict()) if isinstance(self.model, torch.nn.Module) else deepcopy(self.model)

        for client_obj in target_clients:
            try:
                # Ensure client model is on the correct device before loading state_dict
                # This depends on client_obj.model structure and device management
                client_obj.model.load_state_dict(global_model_state)
            except Exception as e:
                self.logger.error(f"Error sending model to client {client_obj.client_idx}: {e}", exc_info=True)


    # MODIFIED train method
    def train(self): 
        training_start_time = time.time()
        self.train_times = getattr(self, 'train_times', []) 
        self.round_times = getattr(self, 'round_times', [])

        for r in range(1, self.global_rounds + 1):
            round_start_time = time.time()
            self.logger.info(f"--- Global Round {r}/{self.global_rounds} ---")

            if self.num_clients == 0:
                self.logger.error("No clients available in the server. Exiting training.")
                break
            if self.model is None:
                self.logger.error("Server model is not initialized. Exiting training.")
                break

            if self.aggregation_strategy == 'fedtop_select_diverse':
                # CRITICAL FIX: Use a configured target number of clients for selection.
                # Assuming `self.num_active_clients` (from ServerBase or args) IS the CONFIGURED INTEGER 
                # for the number of clients to make active per round.
                # If your ServerBase uses a different attribute name for this (e.g., self.clients_per_round_config),
                # use that name here.
                if not self.active_clients:
                    self.logger.warning(f"Round {r}: Topological selection resulted in no active clients. Falling back to standard random sampling for this round.")
                    # Standard random sampling via ServerBase's sample_active_clients()
                    # This method should set self.active_clients to a list of self.num_active_clients (target_selection_count) clients.
                    self.sample_active_clients() 
                    self.send_models()

                self.num_active_clients = len(self.active_clients)
                if not hasattr(self, 'num_active_clients') or not isinstance(self.num_active_clients, int) or self.num_active_clients <= 0:
                    self.logger.error("`self.num_active_clients` is not a valid positive integer. Cannot determine target selection count. Defaulting to 1.")
                    target_selection_count = 1 # Fallback, but should be configured properly
                else:
                    target_selection_count = self.num_active_clients

                num_base_sample_for_candidate_pool = int(self.num_clients * self.sampling_prob) 
                num_candidates = int(num_base_sample_for_candidate_pool * self.candidate_pool_factor)
                
                # Ensure candidate pool is large enough, but not exceeding total clients.
                num_candidates = max(min(num_candidates, self.num_clients), target_selection_count)
                
                self.logger.debug(f"Round {r}: Target selection count: {target_selection_count}. Sampling {num_candidates} candidates for topological selection.")
                
                if num_candidates == 0 and self.num_clients > 0 : # If num_candidates calculation resulted in 0 but clients exist
                    self.logger.warning(f"Round {r}: num_candidates for selection is 0. Will attempt to use target_selection_count ({target_selection_count}) if possible, or fallback.")
                    # This might happen if sampling_prob or candidate_pool_factor is too small or num_clients is small.
                    # We attempt to sample at least target_selection_count if possible.
                    num_candidates_to_sample = min(target_selection_count if target_selection_count > 0 else self.num_clients, self.num_clients)
                    if num_candidates_to_sample == 0 and self.num_clients > 0: # Still zero, try to get at least one if clients exist
                        num_candidates_to_sample = min(1, self.num_clients)

                    if num_candidates_to_sample > 0:
                         candidate_indices = random.sample(range(self.num_clients), num_candidates_to_sample)
                    else:
                         candidate_indices = [] # No clients can be sampled
                elif num_candidates > 0 :
                    candidate_indices = random.sample(range(self.num_clients), num_candidates)
                else: # num_candidates is 0 and self.num_clients is 0
                    candidate_indices = []

                if not candidate_indices:
                    self.logger.warning(f"Round {r}: No candidates could be sampled. Skipping client selection and training for this round.")
                    self.active_clients = []
                else:
                    candidate_clients = [self.clients[i] for i in candidate_indices]
                    self.logger.info(f"Sampled {len(candidate_clients)} candidates for topological selection: {[c.client_idx for c in candidate_clients]}")
                    self.send_models(candidate_clients) 
                    self.active_clients = self.select_clients_topologically(candidate_clients, target_selection_count) # Pass the target count
                 
            else: 
                if r == self.global_rounds and hasattr(self, 'final_round_sampling_prob'): 
                     original_sampling_prob = self.sampling_prob
                     self.sampling_prob = getattr(self, 'final_round_sampling_prob', 1.0)
                     self.sample_active_clients()
                     self.sampling_prob = original_sampling_prob 
                else:
                     self.sample_active_clients() 
                self.send_models()


            if not self.active_clients:
                self.logger.warning(f"Round {r}: No active clients selected/sampled. Skipping training and aggregation for this round.")
                self.round_times.append(time.time() - round_start_time)
                if r % self.eval_gap == 0 or r == self.global_rounds: 
                    self.logger.info(f"Round {r}: Attempting evaluation despite no client training.")
                    self.perform_evaluation(r)
                continue

            client_training_start_time = time.time()
            train_accs, train_losses, client_pds_after_train = self.train_clients_and_get_pds()
            client_training_time = time.time() - client_training_start_time

            if train_accs: 
                avg_train_acc = np.mean(train_accs) 
                avg_train_loss = np.mean(train_losses)
                self.logger.info(f"Round {r}: Avg Client Train Acc: {avg_train_acc:.2f}%, Avg Client Train Loss: {avg_train_loss:.4f}, Client Train Time: {client_training_time:.2f}s")
            else:
                self.logger.warning(f"Round {r}: No clients successfully trained in this round.")


            aggregation_start_time = time.time()
            if hasattr(self, 'updates_from_clients') and self.updates_from_clients: 
                if self.aggregation_strategy == 'fedtopavg':
                    self.reference_pd_for_fedtopavg = self.get_global_model_pd_features() 
                    self.aggregate_models_fedtopavg(client_pds_after_train)
                elif self.aggregation_strategy == 'fedtop_select_diverse' or self.aggregation_strategy == 'fedavg':
                    self.aggregate_models_fedavg()
                else: 
                    self.logger.warning(f"Round {r}: Unknown aggregation strategy '{self.aggregation_strategy}'. Defaulting to FedAvg.")
                    self.aggregate_models_fedavg()
                aggregation_time = time.time() - aggregation_start_time
                self.logger.info(f"Round {r}: Aggregation time: {aggregation_time:.2f}s")
            else:
                self.logger.warning(f"Round {r}: No client updates received. Skipping model aggregation.")
            
            round_duration = time.time() - round_start_time
            self.train_times.append(client_training_time) 
            self.round_times.append(round_duration) 

            self.perform_evaluation(r)

        total_training_time = time.time() - training_start_time
        self.logger.info(f"Total training finished in {total_training_time:.2f} seconds.")
        avg_round_time = np.mean(self.round_times) if self.round_times else 0
        self.logger.info(f"Average round time: {avg_round_time:.2f} seconds.")

    def perform_evaluation(self, round_num):
        if round_num % self.eval_gap == 0 or round_num == self.global_rounds:
            eval_start_time = time.time()
            test_acc, test_loss, test_acc_std = self.evaluate() 
            eval_duration = time.time() - eval_start_time
            self.logger.info(f"Round [{round_num}/{self.global_rounds}] Global Model Eval: Test Acc: {test_acc:.2f}% (std: {test_acc_std:.2f}), Test Loss: {test_loss:.4f}. Eval time: {eval_duration:.2f}s")
            
            if hasattr(self, 'evaluate_personalized') and self.evaluate_personalized: # evaluate_personalized attribute not in class, assume it's a boolean from args/ServerBase
                pers_eval_start_time = time.time()
                # Corrected: Assuming evaluate_personalized is a method name if this block is reached
                # or if `self.evaluate_personalized` is a boolean, it should control calling a method like `self.run_personalized_evaluation()`
                # Given the original code `ptest_acc, ptest_loss, ptest_acc_std = self.evaluate_personalized()`,
                # it implies self.evaluate_personalized is a method.
                # Let's assume this attribute `evaluate_personalized` is a boolean flag.
                # And the actual method to call is, for example, `run_evaluation_personalized`.
                # However, the original had it as self.evaluate_personalized(), so I'll keep that structure,
                # assuming self.evaluate_personalized is either a method or a callable attribute.
                # If it's just a boolean, the calling logic in ServerBase or here needs adjustment.
                # For now, keeping the original call structure.
                ptest_acc, ptest_loss, ptest_acc_std = self.evaluate_personalized() # Changed to avoid conflict if it's a boolean flag
                pers_eval_duration = time.time() - pers_eval_start_time
                self.logger.info(f"Round [{round_num}/{self.global_rounds}] Personalized Model Eval: Test Acc: {ptest_acc:.2f}% (std: {ptest_acc_std:.2f}), Test Loss: {ptest_loss:.4f}. Eval time: {pers_eval_duration:.2f}s")


    def train_clients_and_get_pds(self):
        client_updates_list = [] 
        train_accs = []
        train_losses = []
        client_pds_after_training = {} 

        if not self.active_clients:
            self.logger.warning("train_clients_and_get_pds called with no active clients.")
            return train_accs, train_losses, client_pds_after_training

        global_model_state_dict_for_update = {k: v.cpu() for k, v in self.model.state_dict().items()}

        for client_obj in self.active_clients: 
            self.logger.debug(f"Training client {client_obj.client_idx}...")
            try:
                acc, loss = client_obj.train() 
                if np.isnan(acc) or np.isinf(acc) or np.isnan(loss) or np.isinf(loss) or loss == float('inf'):
                    self.logger.warning(f"Client {client_obj.client_idx} training returned invalid metrics. Acc: {acc}, Loss: {loss}. Skipping update and PD from this client.")
                    continue 
                train_accs.append(acc)
                train_losses.append(loss)
 
                update = client_obj.get_update(global_model_state_dict_for_update) 
                
                num_train_samples = getattr(client_obj, 'num_train', 0)
                if num_train_samples <= 0:
                    self.logger.warning(f"Client {client_obj.client_idx} reported {num_train_samples} training samples. Update will have 0 weight if data-based weighting is used without topo adjustment.")
                
                if update is not None and update.numel() > 0:
                     client_updates_list.append({'client_idx': client_obj.client_idx, 'update': update, 'num_samples': max(num_train_samples, 0)})
                else:
                    self.logger.warning(f"Client {client_obj.client_idx} provided None or empty update after training. Excluding from aggregation.")
                    continue 

                if self.aggregation_strategy == 'fedtopavg':
                    pd_info = client_obj.get_pd_features() 
                    if pd_info and not pd_info.get('error', True) and pd_info.get('diagrams') is not None:
                        client_pds_after_training[client_obj.client_idx] = pd_info['diagrams']
                        self.logger.debug(f"Successfully got post-training PD for client {client_obj.client_idx}.")
                    else:
                        self.logger.warning(f"Could not get post-training PD for client {client_obj.client_idx} for FedTopAvg. PD_info: {pd_info}")
                        client_pds_after_training[client_obj.client_idx] = None 

            except Exception as e:
                self.logger.error(f"Error during training or PD generation for client {client_obj.client_idx}: {e}\n{traceback.format_exc()}")
        
        self.updates_from_clients = client_updates_list 
        return train_accs, train_losses, client_pds_after_training


    def aggregate_models_fedavg(self):
        self.logger.info("Aggregating models using FedAvg.")
        if not hasattr(self, 'updates_from_clients') or not self.updates_from_clients:
            self.logger.warning("No client updates to aggregate for FedAvg. Global model remains unchanged.")
            return

        aggregated_deltas = {name: torch.zeros_like(param, device='cpu') for name, param in self.model.named_parameters()}
        total_samples_in_round = sum(item['num_samples'] for item in self.updates_from_clients if item['num_samples'] > 0)
        
        if total_samples_in_round == 0:
            self.logger.warning("Total samples for FedAvg aggregation is zero. No weighted aggregation possible. Model not updated unless there's only one update (unweighted average).")
            if len(self.updates_from_clients) > 0:
                 self.logger.info("Applying unweighted average as total_samples_in_round is 0 but updates exist.")
                 num_valid_updates_unweighted = 0
                 for item in self.updates_from_clients:
                    full_update_vector = item['update'].to('cpu')
                    if full_update_vector.numel() == 0: continue
                    num_valid_updates_unweighted +=1
                    current_pos = 0
                    for name, param in self.model.named_parameters():
                        num_elements = param.numel()
                        if current_pos + num_elements > full_update_vector.numel(): break 
                        delta_slice = full_update_vector[current_pos : current_pos + num_elements].view_as(param)
                        aggregated_deltas[name].add_(delta_slice)
                        current_pos += num_elements
                 if num_valid_updates_unweighted > 0:
                     for name in aggregated_deltas:
                         aggregated_deltas[name].div_(num_valid_updates_unweighted)
                 else:
                     self.logger.warning("No valid updates for unweighted average either.")
                     return       
            else: 
                return

        else: 
            num_valid_updates_fedavg = 0
            for item in self.updates_from_clients:
                if item['num_samples'] <= 0: continue 
                num_valid_updates_fedavg +=1
                weight = item['num_samples'] / total_samples_in_round
                full_update_vector = item['update'].to('cpu') 
                current_pos = 0
                client_update_valid = True
                for name, param in self.model.named_parameters(): 
                    num_elements = param.numel()
                    if current_pos + num_elements > full_update_vector.numel():
                        self.logger.error(f"FedAvg: Update vector from client {item['client_idx']} (len {full_update_vector.numel()}) is too short for param {name} (needs {num_elements} at pos {current_pos}). Skipping this client's contribution.")
                        client_update_valid = False
                        break 
                    try:
                        delta_slice = full_update_vector[current_pos : current_pos + num_elements].view(param.shape) 
                        aggregated_deltas[name].add_(delta_slice * weight) 
                    except RuntimeError as e:
                        self.logger.error(f"FedAvg: Error reshaping/adding delta for param {name} from client {item['client_idx']}. Slice shape {delta_slice.shape if 'delta_slice' in locals() else 'N/A'}, Param shape {param.shape}. Error: {e}. Skipping client.")
                        client_update_valid = False
                        break 
                    current_pos += num_elements
                
                if client_update_valid and current_pos != full_update_vector.numel():
                     self.logger.warning(f"FedAvg: Update vector from client {item['client_idx']} was not fully consumed. Consumed {current_pos}, total {full_update_vector.numel()}. Possible model structure mismatch.")
            
            if num_valid_updates_fedavg == 0:
                 self.logger.warning("No valid client updates (with positive samples) were processed in FedAvg. Model not updated by weighted average.")
                 return

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.add_(aggregated_deltas[name].to(param.device)) 

        self.logger.info("FedAvg aggregation complete.")


    def aggregate_models_fedtopavg(self, client_pds_after_training):
        self.logger.info("Aggregating models using FedTopAvg.")
        if not hasattr(self, 'updates_from_clients') or not self.updates_from_clients:
            self.logger.warning("No client updates for FedTopAvg. Global model remains unchanged.")
            return

        if self.reference_pd_for_fedtopavg is None:
            self.logger.warning("Reference PD (global model PD before training) is None for FedTopAvg. Falling back to standard FedAvg.")
            self.aggregate_models_fedavg()
            return

        topological_similarities_raw = {} 
        for item in self.updates_from_clients:
            client_idx = item['client_idx']
            client_pd_after_train = client_pds_after_training.get(client_idx) 
            
            if client_pd_after_train is None: 
                self.logger.debug(f"Client {client_idx} has no post-training PD for FedTopAvg similarity. Assigning zero similarity.")
                topological_similarities_raw[client_idx] = 0.0
                continue

            dist = self.pd_distance(client_pd_after_train, self.reference_pd_for_fedtopavg, self.tda_homology_dim_to_use)
            
            if np.isinf(dist) or np.isnan(dist) or dist < 0: 
                similarity = 0.0
                self.logger.debug(f"Client {client_idx}: PD distance to global is {dist}. Similarity set to 0.")
            else:
                similarity = np.exp(-dist / self.fedtopavg_similarity_scale) 
                self.logger.debug(f"Client {client_idx}: PD distance to global pre-train model is {dist:.4f}, raw similarity {similarity:.4f}.")
            topological_similarities_raw[client_idx] = similarity

        sum_raw_similarities = sum(topological_similarities_raw.values())
        normalized_topo_weights = {} 
        if sum_raw_similarities > 1e-9: 
            for client_idx, sim in topological_similarities_raw.items():
                normalized_topo_weights[client_idx] = sim / sum_raw_similarities
        else: 
            self.logger.warning("Sum of raw topological similarities is near zero. Topological weights will be uniform if any clients contributed.")
            num_clients_with_updates = len(self.updates_from_clients)
            if num_clients_with_updates > 0:
                for item in self.updates_from_clients:
                     normalized_topo_weights[item['client_idx']] = 1.0 / num_clients_with_updates
            else: 
                 pass

        final_combined_weights = {} 
        total_samples_in_round = sum(item['num_samples'] for item in self.updates_from_clients if item['num_samples'] > 0)

        for item in self.updates_from_clients:
            client_idx = item['client_idx']
            data_weight = 0.0
            if total_samples_in_round > 0 and item['num_samples'] > 0:
                data_weight = item['num_samples'] / total_samples_in_round
            
            topo_w = normalized_topo_weights.get(client_idx, 0.0) 
            final_combined_weights[client_idx] = (1 - self.topo_weight_lambda) * data_weight + self.topo_weight_lambda * topo_w

        sum_final_combined_weights = sum(final_combined_weights.values())
        if sum_final_combined_weights > 1e-9:
            for client_idx in final_combined_weights:
                final_combined_weights[client_idx] /= sum_final_combined_weights
        elif total_samples_in_round > 0 : 
            self.logger.warning("FedTopAvg: Sum of final combined weights is zero. Falling back to pure data weights as samples exist.")
            for item in self.updates_from_clients:
                final_combined_weights[item['client_idx']] = (item['num_samples'] / total_samples_in_round) if item['num_samples'] > 0 else 0.0
        elif len(self.updates_from_clients) > 0: 
            self.logger.warning("FedTopAvg: Sum of final combined weights and total samples are zero. Falling back to uniform weights over available updates.")
            num_updates = len(self.updates_from_clients)
            for item in self.updates_from_clients:
                final_combined_weights[item['client_idx']] = 1.0 / num_updates
        else: 
            self.logger.warning("FedTopAvg: No updates to aggregate.")
            return


        self.logger.info(f"FedTopAvg final client weights (sum {sum(final_combined_weights.values()):.4f}):")
        for idx, w in sorted(final_combined_weights.items(), key=lambda x: x[0]):
            if w > 1e-5: self.logger.info(f"  Client {idx}: {w:.4f}")

        aggregated_deltas = {name: torch.zeros_like(param, device='cpu') for name, param in self.model.named_parameters()}
        num_valid_updates_fedtop = 0
        for item in self.updates_from_clients:
            client_idx = item['client_idx']
            weight = final_combined_weights.get(client_idx, 0.0)
            
            if weight <= 1e-9 : continue 
            num_valid_updates_fedtop +=1

            full_update_vector = item['update'].to('cpu')
            current_pos = 0
            client_update_valid = True
            for name, param in self.model.named_parameters():
                num_elements = param.numel()
                if current_pos + num_elements > full_update_vector.numel():
                    self.logger.error(f"FedTopAvg: Update vector from client {client_idx} too short for param {name}. Skipping this client's weighted contribution.")
                    client_update_valid = False
                    break
                try:
                    delta_slice = full_update_vector[current_pos : current_pos + num_elements].view(param.shape)
                    aggregated_deltas[name].add_(delta_slice * weight)
                except RuntimeError as e:
                    self.logger.error(f"FedTopAvg: Error reshaping/adding delta for param {name} from client {item['client_idx']}. Error: {e}. Skipping client.")
                    client_update_valid = False
                    break
                current_pos += num_elements
            
            if client_update_valid and current_pos != full_update_vector.numel():
                 self.logger.warning(f"FedTopAvg: Update vector from client {client_idx} not fully consumed. Consumed {current_pos}, total {full_update_vector.numel()}.")

        if num_valid_updates_fedtop == 0:
            self.logger.warning("No valid client updates were effectively weighted in FedTopAvg. Model not updated.")
            return

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.add_(aggregated_deltas[name].to(param.device))

        self.logger.info("FedTopAvg aggregation complete.")

    def evaluate(self):
        total_samples = sum(c.num_test for c in self.clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        for c in self.clients:
            old_model = deepcopy(c.model)
            c.model = deepcopy(self.model)
            acc, loss = c.evaluate()
            accs.append(acc)
            weighted_loss += (c.num_test / total_samples) * loss.detach()
            weighted_acc += (c.num_test / total_samples) * acc
            c.model = old_model
        std = torch.std(torch.stack(accs))
        return weighted_acc, weighted_loss, std

    def evaluate_personalized(self):
        total_samples = sum(c.num_test for c in self.clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        for c in self.clients:
            old_model = deepcopy(c.model)
            c.model = deepcopy(self.model)
            c.train()
            acc, loss = c.evaluate()
            accs.append(acc)
            weighted_loss += (c.num_test / total_samples) * loss.detach()
            weighted_acc += (c.num_test / total_samples) * acc
            c.model = old_model
        std = torch.std(torch.stack(accs))
        return weighted_acc, weighted_loss, std