import sys 
sys.path.append('../..')
import numpy as np
import torch
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# A simple env to test one-step RL algorithms, based on the difference of two signals
class Environment():
    def __init__(self, targets, state, surrogate, savety_check=False): 
        self.savety_check = savety_check
        # Set surrogate
        self.surrogate = surrogate
        # Set target shape
        self.target_shape = targets[0].shape[-1]
    
        # Set state
        self.init_sys_state = state[0]
        self.state = self.init_sys_state.clone()
        
        # Set target
        self.targets = targets
        #self.target = self.targets[np.newaxis,np.random.randint(0,len(self.targets))]
        self.target = self.targets[np.random.randint(0,len(self.targets))]
        
        # Set sizes
        self.action_space = self.target_shape
        self.observation_space = self.target_shape*2
        
    def step(self, action):
        if self.savety_check:
            if not eval_signal(action[0,1:].cpu().detach().numpy()):
                self.state = -1 * torch.ones(self.state.shape[0]).float().to(device)
                return
        # Perform one step
        with torch.no_grad():
            surrogate_state = torch.cat([self.init_sys_state.unsqueeze(0), action],dim=1)
            self.state = self.surrogate.model(surrogate_state)[0]
        #self.state = torch.clip(delta_action[0] + self.state, 0, 1)
        #return state, reward, done, _
    
    def reset(self):
        # Reset the env
        self.state = self.init_sys_state.clone()
        #self.action_state = self.init_action_state.clone()
        self.target = self.targets[np.random.randint(0,len(self.targets))]
        #return state