from abc import ABC
import numpy as np
import torch
import logging
from copy import deepcopy

class Server(ABC):
    def __init__(self, args):
        self.num_clients = args.num_clients
        self.model = deepcopy(args.model)
        self.global_rounds = args.global_rounds
        self.clients = []
        try:
            self.D = self.model.D  # Try to get D from model
        except AttributeError:
            # If model doesn't have D attribute, set a default value or calculate it
            if hasattr(self.model, 'feature_dim'):
                self.D = self.model.feature_dim
            else:
                # Set a reasonable default based on your model architecture
                self.D = 512  # Common feature dimension size
                print(f"Warning: Model {args.model_name} has no 'D' attribute. Using default value {self.D}.")
        self.num_classes = args.num_classes
        self.device = args.device
        self.active_clients = []
        self.num_active_clients = 0
        self.sampling_prob = args.sampling_prob
        self.active_client_ids = []
        self.eval_gap = args.eval_gap
        self.train_times = []
        self.round_times = []
        logger_name = getattr(self, 'id', 'Server_DefaultID')
        self.logger = logging.getLogger(str(logger_name))
        if not self.logger.hasHandlers(): 
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)
        self.logger.setLevel(getattr(logging, 'INFO', logging.INFO))

    def send_models(self):
        for c in self.active_clients:
            c.set_model(self.model)

    def sample_active_clients(self):
        self.active_clients = []
        self.active_client_ids = []
        sampling_prob_tensor = torch.ones(self.num_clients)*self.sampling_prob
        selected_indices = (torch.bernoulli(sampling_prob_tensor).numpy()).astype(int)
        selected_indices = np.where(selected_indices==1)[0]
        self.selected_clients_indices = list(np.unique(selected_indices))
        for idx in self.selected_clients_indices:
            self.active_clients.append(self.clients[idx])
            self.active_client_ids.append(idx)
        self.num_active_clients = len(self.active_clients)

    def aggregate_models(self):
        total_samples = sum(c.num_train for c in self.active_clients)
    
        for param in self.model.parameters():
            param.data.zero_()

        for c in self.active_clients:
            for global_param, client_param in zip(self.model.parameters(), c.model.parameters()):
                client_data = client_param.data.to(global_param.device)
                global_param.data = global_param.data + (c.num_train / total_samples)* client_data

    def evaluate(self, only_active=False):
        clients = self.active_clients if only_active else self.active_clients
        total_samples = sum(c.num_test for c in clients)
        weighted_loss = 0
        weighted_acc = 0
        accs = []
        for c in clients:
            acc, loss = c.evaluate()
            accs.append(acc)
            weighted_loss += (c.num_test / total_samples) * loss.detach()
            weighted_acc += (c.num_test / total_samples) * acc
        std = torch.std(torch.stack(accs))
        return weighted_acc, weighted_loss, std
    
    def evaluate_personalized(self):
        pass

    def train_clients(self):
        train_acc, train_loss = 0, 0
        num_total = sum(c.num_train for c in self.active_clients)
        for i, c in enumerate(self.active_clients):
            client_acc, client_loss = c.train()
            train_acc += (c.num_train / num_total) * client_acc
            train_loss += (c.num_train / num_total) * client_loss
        return train_acc, train_loss
