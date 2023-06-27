import numpy as np
import torch
from mlp import *
from buffer import *
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# A simple surrogate
class SimpleSurrogate():
    def __init__(self, target_size, hidden_size=None, num_hidden=1, activation=torch.tanh, lr=1e-4, loss_f=torch.nn.MSELoss(), last_layer_activation=True, input_size=None):
        hidden_size = target_size if hidden_size == None else hidden_size
        # Model
        if input_size==None:
            self.model = MLP(target_size*2, target_size, hidden_size, num_hidden, activation = activation, last_layer_activation=last_layer_activation).to(device)
        else:
            self.model = MLP(input_size, target_size, hidden_size, num_hidden, activation = activation, last_layer_activation=last_layer_activation).to(device)
        # Experience buffer
        self.experiences = Buffer()
        
        # Optimization 
        self.optim = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.loss_f = loss_f
        
    # Offline training with a data loader
    def train(self, train_loader, num_epochs=1000):
        # Loss track
        loss_evol = []
        # Pretrain
        pbar = tqdm(range(num_epochs))
        # Epochs
        for epoch in pbar:
            loss_epoch = []
            # Iterations
            for env_input, env_output in train_loader:
                self.optim.zero_grad()
                predicted_state = self.model(env_input)
                loss = self.loss_f(predicted_state, env_output)
                loss_epoch.append(loss.item())
                loss.backward()
                self.optim.step()
            train_loss = np.mean(loss_epoch)
            loss_evol.append(train_loss)
            pbar.set_postfix({'Train loss': train_loss})      
        return(np.array(loss_evol))