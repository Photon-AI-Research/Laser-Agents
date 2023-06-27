import torch
import torch.nn as nn
import sys
sys.path.append('../..')
from utils import *
import torchpwl

# Simple MLP model with normalization features
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden, activation = torch.relu, last_layer_activation = True, norm=False, clip_norm=False, minmax_norm = True, pwl=False, xn=None, xp=None):
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
        
        self.pwl = pwl
        # Points to infer
        self.xn = xn
        # Knots - shape must match dim of output size
        self.xp = xp
    
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
        
    def forward(self, x, pwl=False, xp=None, yp=None):
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
        if self.pwl==False:
            return x
        else:
            
            xp_batch = self.xp.unsqueeze(0).repeat(x.shape[0],1).to(x.device)
            x_batch = self.xn.unsqueeze(1).repeat(1,x.shape[0]).to(x.device)

            out = torchpwl.calibrate1d(x_batch, xp_batch, x[:,1:]).T
            power_out = torch.cat([x[:,:1],out], dim=1)
            return power_out