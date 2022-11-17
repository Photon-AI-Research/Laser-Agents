import torch
import torch.nn as nn
import sys
sys.path.append('../..')
from utils import *

# Simple MLP model with normalization features
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden, activation = torch.relu, last_layer_activation = True, norm=False, clip_norm=False, minmax_norm = True):
        super(MLP, self).__init__()
        torch.manual_seed(1234)
        self.linear_layers = nn.ModuleList()
        self.activation = activation
        self.init_layers(input_size, output_size, hidden_size,num_hidden)
        # Turn on activation on last layer
        self.last_layer_activation = last_layer_activation
        # Normalization (experimental)
        self.norm = norm
        self.clip_norm= clip_norm
        self.minmax_norm = minmax_norm
    
    # Initialize model weights
    def init_layers(self, input_size, output_size, hidden_size, num_hidden):
        torch.manual_seed(1234)
        if num_hidden == 0:
            self.linear_layers.append(nn.Linear(input_size, output_size))
        else:
            self.linear_layers.append(nn.Linear(input_size, hidden_size))
            for _ in range(num_hidden):
                self.linear_layers.append(nn.Linear(hidden_size, hidden_size))
            self.linear_layers.append(nn.Linear(hidden_size, output_size))

        for m in self.linear_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        for i in range(0, len(self.linear_layers) - 1):
            x = self.linear_layers[i](x)
            x = self.activation(x)
        x = self.linear_layers[len(self.linear_layers) - 1](x)
        if self.last_layer_activation:
            x = self.activation(x)
        
        # Normalization (experimental)
        if self.norm and self.minmax_norm:
            if self.clip_norm:
                x = torch.clip(x, 0, 1)
            x = torch.cat([x[:,0].unsqueeze(1),norm_tensor(x[:,1:], dim=-1)],dim=-1)
        if self.norm and not self.minmax_norm:
            x = torch.cat([x[:,0].unsqueeze(1),norm_tensor(x[:,1:], dim=-1, minmax_norm=self.minmax_norm)],dim=-1)
            if self.clip_norm:
                x = torch.clip(x, 0, 1)
        return x