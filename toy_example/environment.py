import sys 
sys.path.append('../..')
import numpy as np
import torch
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# A simple env to test one-step RL algorithms, based on the difference of two signals
class Environment():
    def __init__(self, targets): 
        # Set target shape
        self.target_shape = targets[0].shape[-1]
    
        # Set state
        self.init_sys_state = gauss_range(self.target_shape).to(device)
        self.state = self.init_sys_state.clone().to(device)
        
        # set action
        #self.init_action_state = torch.ones(self.target_shape).to(device)/2
        #self.action_state = self.init_action_state.clone().to(device)
        
        # Set target
        self.targets = targets.to(device)
        self.target = self.targets[np.random.randint(0,len(self.targets))]
        self.goal_action = self.target-self.init_sys_state
        
        # Set sizes
        self.action_space = self.target_shape
        self.observation_space = self.target_shape*2
        
    def step(self, delta_action):
        # Perform one step
        #self.action_state = torch.clip(self.action_state + delta_action, -1, 1)
        #self.state = torch.clip(delta_action[0] + self.init_sys_state, 0, 1)
        self.state = torch.clip(delta_action[0] + self.state, 0, 1)
        #return state, reward, done, _
    
    def reset(self):
        # Reset the env
        self.state = self.init_sys_state.clone()
        #self.action_state = self.init_action_state.clone()
        self.target = self.targets[np.random.randint(0,len(self.targets))]
        self.goal_action = self.target-self.init_sys_state
        #return state