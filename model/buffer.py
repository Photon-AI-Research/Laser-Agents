import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# A simple experience Buffer
class Buffer():
    def __init__(self, size = 50000):
        # Initialize memory tensors
        self.state_memory = torch.tensor([]).to(device)
        self.action_memory = torch.tensor([]).to(device)
        self.next_state_memory = torch.tensor([]).to(device)
        self.org_target_memory = torch.tensor([]).to(device)
        self.size = size
        
        # Experience that is always added to the sample
        self.vip_state_memory = torch.tensor([]).to(device)
        self.vip_action_memory = torch.tensor([]).to(device)
        self.vip_next_state_memory = torch.tensor([]).to(device)
        self.vip_org_target_memory = torch.tensor([]).to(device)
    
    # Add experience to buffer
    def add(self, state, action, next_state, org_target):
        # Add to memory tensors
        self.state_memory = torch.cat([self.state_memory, state.unsqueeze(0)])
        self.action_memory = torch.cat([self.action_memory, action.unsqueeze(0)])
        self.next_state_memory = torch.cat([self.next_state_memory, next_state.unsqueeze(0)])
        self.org_target_memory = torch.cat([self.org_target_memory, org_target.unsqueeze(0)])
        # If the current size exceeeds the buffer size keep only recent
        if len(self.state_memory) > self.size:
            self.state_memory = self.state_memory[-self.size:]
            self.action_memory  =self.action_memory [-self.size:]
            self.next_state_memory = self.next_state_memory[-self.size:]
            self.org_target_memory = self.org_target_memory[-self.size:]
            
    def sample(self, batch_size, prior=True, alpha=0.4):
        #idx = np.random.randint(0,len(self.state_memory),size=batch_size)
        idx = np.arange(len(self.state_memory))
        # If prior is true recent samples have heigher priority
        w = (idx**alpha)/np.linalg.norm(idx**alpha,1) if prior else np.ones(len(idx))/np.linalg.norm(np.ones(len(idx)),1)
        batch_idx = np.random.choice(idx, batch_size, p=w)
        if self.vip_state_memory.shape[0] == 0:
            # Return memory
            return self.state_memory[batch_idx], self.action_memory[batch_idx], self.next_state_memory[batch_idx], self.org_target_memory[batch_idx]
        else:
            # Return memory mixed with vip samples
            mix_states = torch.cat([self.vip_state_memory,self.state_memory[batch_idx]])
            mix_actions = torch.cat([self.vip_action_memory,self.action_memory[batch_idx]])
            mix_next_states = torch.cat([self.vip_next_state_memory,self.next_state_memory[batch_idx]])
            mix_org_targets = torch.cat([self.vip_org_target_memory,self.org_target_memory[batch_idx]])
            return mix_states, mix_actions, mix_next_states, mix_org_targets
    
    def sample_all(self):
        return self.state_memory, self.action_memory, self.next_state_memory, self.org_target_memory