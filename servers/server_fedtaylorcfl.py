# File: servers/server_fedtaylorcfl.py
import time
import torch
from copy import deepcopy
from collections import OrderedDict
from servers.server_base import Server # Assuming your ServerFedAvg inherits from this
from clients.client_fedtaylorcfl import ClientFedTaylorCFL
from models.cnn import CIFARNetTaylor # Import the new model

class ServerFedTaylorCFL(Server):
    def __init__(self, args):
        super().__init__(args) # Initializes self.model (base model, e.g. from args.model_name)
        
        # Override self.model with CIFARNetTaylor
        # You might need to pass D_h_taylor, N_taylor, taylor_activation from args
        self.model = CIFARNetTaylor(
            num_classes=args.num_classes, 
            in_channels=args.in_channels,
            D_h_taylor=args.D_h_taylor, # e.g., 128 or 64
            N_taylor=args.N_taylor,     # e.g., 4 or 8
            taylor_activation=args.taylor_activation # e.g., "gelu"
        ).to(args.device) # Keep global model on device if server does computations

        self.clients = []
        for client_idx in range(self.num_clients):
            # Pass is_corrupted status if you have it
            is_corrupted = False # Placeholder
            c = ClientFedTaylorCFL(args, client_idx, is_corrupted)
            self.clients.append(c)
        
        # Ensure Server base class has these attributes if used in logging
        self.train_times = []
        self.round_times = []


    def send_models(self):
        """Send the current global model (CIFARNetTaylor state_dict) to active clients."""
        for c in self.active_clients:
            # Client expects a model object, not just state_dict for its c.model
            c.set_model(deepcopy(self.model).to("cpu")) # Send a CPU copy

    def aggregate_models(self):
        """
        Averages the state_dicts received from active clients.
        Assumes self.client_uploads contains the state_dicts from clients.
        """
        if not self.client_uploads:
            self.logger.info("Warning: No client uploads to aggregate.")
            return

        # Initialize a new global_state_dict with zeros
        global_state_dict = OrderedDict()
        for k in self.model.state_dict().keys():
            global_state_dict[k] = torch.zeros_like(self.model.state_dict()[k], dtype=torch.float32, device="cpu")
        
        num_active_clients = len(self.client_uploads)
        
        for client_state_dict in self.client_uploads:
            for k in global_state_dict.keys():
                if k in client_state_dict:
                    global_state_dict[k] += client_state_dict[k].to(torch.float32) # Ensure float for accumulation
                else:
                    self.logger.info(f"Warning: Key {k} not found in client upload during aggregation.")

        for k in global_state_dict.keys():
            global_state_dict[k] /= num_active_clients
            
        self.model.load_state_dict(global_state_dict)
        self.model = self.model.to(self.device) # Move back to server's device

    def train_clients(self):
        """
        Handles training on active clients and collecting their parameters.
        (This method is often part of ServerBase or the main train loop)
        """
        client_params_list = []
        total_loss = 0
        total_acc = 0
        num_samples = 0

        for c in self.active_clients:
            acc, loss = c.train() # Local training
            client_params_list.append(c.get_parameters_to_send()) # Get updated state_dict
            
            # For logging average train loss/acc (weighted by client data samples if available)
            # Assuming c.num_train_samples exists
            client_samples = getattr(c, 'num_train_samples', 1) # default to 1 if not available
            total_loss += loss * client_samples
            total_acc += acc * client_samples
            num_samples += client_samples
        
        self.client_uploads = client_params_list # Store for aggregate_models

        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        avg_acc = total_acc / num_samples if num_samples > 0 else 0
        return avg_acc, avg_loss

    # train() method would be similar to ServerFedAvg, but calls self.aggregate_models()
    # which now handles state_dicts.
    # evaluate() and evaluate_personalized() from your ServerFedAvg might largely remain
    # the same as they operate on the `self.model` which is now CIFARNetTaylor.

    def train(self): # Example main training loop for server
        for r in range(1, self.global_rounds + 1):
            start_time = time.time()
            
            self.sample_active_clients() # From ServerBase probably
            self.send_models()

            # Train clients and collect their parameters (state_dicts)
            train_acc, train_loss = self.train_clients() 
            train_time = time.time() - start_time
            
            # Aggregate the collected state_dicts
            self.aggregate_models()

            round_time = time.time() - start_time
            self.train_times.append(train_time)
            self.round_times.append(round_time)

            # Logging (similar to your ServerFedAvg)
            if r % self.eval_gap == 0 or r == self.global_rounds:
                # Ensure evaluate methods correctly handle the new model type if needed
                # But deepcopying self.model should work
                ptest_acc, ptest_loss, ptest_acc_std = self.evaluate_personalized()  
                test_acc, test_loss, test_acc_std = self.evaluate() 
                self.logger.info(f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}]\t Test Loss [{test_loss:.4f}|{ptest_loss:.4f}]\t Test Acc [{test_acc:.2f}({test_acc_std:.2f})|{ptest_acc:.2f}({ptest_acc_std:.2f})]\t Round Time [{round_time:.2f}]")
            else:
                self.logger.info(f"Round [{r}/{self.global_rounds}]\t Train Loss [{train_loss:.4f}]\t Train Acc [{train_acc:.2f}]\t Round Time [{round_time:.2f}]")

    # evaluate and evaluate_personalized methods from your ServerFedAvg should mostly work,
    # as they deepcopy self.model and pass it to clients. Ensure client's evaluate
    # method is compatible.
    def evaluate(self): # From your ServerFedAvg, should be mostly compatible
        total_samples = sum(getattr(c, 'num_test_samples', c.num_test) for c in self.clients) # Ensure attribute exists
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        # self.model should be on the correct device (args.device)
        # Client's evaluate method will move it to its device
        for c in self.clients:
            # Store client's current local model if it has one and you need to restore it
            # For evaluation, client just needs the global model.
            original_client_model_state = deepcopy(c.model.state_dict())

            c.set_model(deepcopy(self.model).to("cpu")) # Give client a CPU copy
            acc, loss = c.evaluate() # Client's evaluate method
            accs.append(acc)
            
            # Weighted average
            client_eval_samples = getattr(c, 'num_test_samples', c.num_test)
            weighted_loss += (client_eval_samples / total_samples) * loss # loss should be a scalar tensor or float
            weighted_acc += (client_eval_samples / total_samples) * acc

            # Restore client's original model state if necessary
            c.model.load_state_dict(original_client_model_state)
            c.model = c.model.to("cpu") # Or client's original device
            
        std = torch.std(torch.tensor(accs)) if accs else torch.tensor(0.0)
        return weighted_acc, weighted_loss, std

    def evaluate_personalized(self): # From your ServerFedAvg
        total_samples = sum(getattr(c, 'num_test_samples', c.num_test) for c in self.clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        for c in self.clients:
            original_client_model_state = deepcopy(c.model.state_dict())

            c.set_model(deepcopy(self.model).to("cpu")) # Give client global model
            c.train() # Personalize by training one more time (or few steps)
            acc, loss = c.evaluate() # Evaluate personalized model
            accs.append(acc)

            client_eval_samples = getattr(c, 'num_test_samples', c.num_test)
            weighted_loss += (client_eval_samples / total_samples) * loss
            weighted_acc += (client_eval_samples / total_samples) * acc
            
            c.model.load_state_dict(original_client_model_state)
            c.model = c.model.to("cpu")

        std = torch.std(torch.tensor(accs)) if accs else torch.tensor(0.0)
        return weighted_acc, weighted_loss, std