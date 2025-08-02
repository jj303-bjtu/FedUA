# File: clients/client_fedtaylorcfl.py
import torch
from collections import OrderedDict
from clients.client_base import Client # Assuming your ClientFedAvg inherits from this
from utils.util import AverageMeter # Assuming this is your AverageMeter

class ClientFedTaylorCFL(Client):
    def __init__(self, args, client_idx, is_corrupted=False): # Added default for is_corrupted
        super().__init__(args, client_idx, is_corrupted)
        # self.model is initialized in Server with CIFARNetTaylor
        self.lr = args.lr 
        self.momentum = args.momentum
        self.wd = args.wd
        self.local_epochs = args.local_epochs
        self.device = args.device
        # self.loss is likely CrossEntropyLoss, inherited or set

    def train(self):
        """
        Train the self.model (CIFARNetTaylor) locally.
        """
        trainloader = self.load_train_data()
        # Important: Optimizer should get parameters of CIFARNetTaylor
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wd)
        
        self.model = self.model.to(self.device)
        self.model.train()
        
        losses = AverageMeter()
        accs = AverageMeter()

        for e in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                loss = self.loss(output, y)

                acc = (output.argmax(1) == y).float().mean() * 100.0
                accs.update(acc.item(), x.size(0))
                losses.update(loss.item(), x.size(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        self.model = self.model.to("cpu") # Move model to CPU before sending params
        return accs.avg, losses.avg

    def get_parameters_to_send(self):
        """
        Returns the state dictionary of the locally trained model.
        The server will then average these state dictionaries.
        """
        self.model.eval() # Ensure model is in eval mode
        return OrderedDict((k, v.cpu().clone()) for k, v in self.model.state_dict().items())

    # set_model, evaluate, load_train_data, load_test_data would be similar to ClientFedAvg
    # or inherited from ClientBase