import torch
import torch.nn as nn

class MLP_Conv(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden, kernel_size=7, activation = torch.relu):
        super(MLP_Conv, self).__init__()
        torch.manual_seed(1234)
        self.linear_layers = nn.ModuleList()
        self.activation = activation
        self.init_layers(input_size, output_size, hidden_size,num_hidden, kernel_size)
    
    
    def init_layers(self, input_size, output_size, hidden_size, num_hidden, kernel_size):
        torch.manual_seed(1234)
        if num_hidden == 0:
            self.linear_layers.append(nn.Linear(input_size, output_size))
        else:
            self.linear_layers.append(nn.Linear(input_size, hidden_size))
            for _ in range(num_hidden):
                self.linear_layers.append(nn.Linear(hidden_size, hidden_size))
            self.linear_layers.append(nn.Linear(hidden_size, output_size))
        self.linear_layers.append(torch.nn.Conv1d(1,1, kernel_size=kernel_size, padding=kernel_size//2))
        for m in self.linear_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        for i in range(0, len(self.linear_layers) - 1):
            x = self.linear_layers[i](x)
            x = self.activation(x)
        x = self.linear_layers[len(self.linear_layers) - 1](x.unsqueeze(1).squeeze(0)).squeeze(1)
        x = self.activation(x)
        return x