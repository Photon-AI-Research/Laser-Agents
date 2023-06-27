import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
sys.path.append('../rl_tests/additive_rework/')
from mlp import *
from old_mlp import OldMLP
from tqdm import tqdm
import itertools
from utils import *
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from simple_agent import *
from simple_surrogate import *
from buffer import *
from surrogate_environment import *
import scipy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)


surrogate_num_hidden = 1
agent_num_hidden = 1
# Optimization learning rates
agent_lr = 1e-5
surrogate_lr = 1e-5

# Count number of steps performed (incremented at wave saving)
setting_counter =0

# Plot generated wave
plot = True

# Initalize seperate Buffer, agent external (testing)
experiences = Buffer(50000)

smooth_states = True
norm_states = True
kernel_size = 25

# Fix power to a default value
fixed_power = False
fixed_power_value = 0.03

#Normalize output
norm_output = False

# Smoothing of agent output
smooth_action = False
kernel_size_action = 21
smooth_action_optim = False # Does currently not work

# Normalization as agent feature
norm_feature = False
clip_norm = False
minmax_norm=False

# Activation function in last layer
last_layer_act_surrogate=True

x_vals_setting = np.load('../setting_x_values.npy')
x_vals_spectrum = np.load('../spectrum_x_values.npy')


pre_states, pre_actions = get_data(smooth_states=smooth_states, norm_states = norm_states, train_with_live_data=False)
pre_start_states = pre_states[:,:int(pre_states.shape[1]/2)]
pre_next_states = pre_states[:,int(pre_states.shape[1]/2):]
pre_surrogate_states = torch.cat([pre_start_states, pre_actions], dim=1)
pre_surrogate_actions = pre_next_states

states, actions = get_data(smooth_states=smooth_states, norm_states = norm_states)
start_states = states[:,:int(states.shape[1]/2)]
next_states = states[:,int(states.shape[1]/2):]
surrogate_states = torch.cat([start_states, actions], dim=1)
surrogate_actions = next_states

max_sample_day_org_idx = np.argmax(np.unique(states[:,:2000], axis=0, return_counts=True)[1])
max_sample_day_start = np.unique(states[:,:2000], axis=0)[max_sample_day_org_idx]
max_sample_day_idx_list = np.where(np.isclose(states[:,:2000], max_sample_day_start).sum(axis=1)==2000)[0]

last_layer_act_agent=True
agent_num_hidden = 1
surrogate_num_hidden_ = 1
#agent_name = "data_fix_512_1_20000000_1oxgpxiz_vermilion-springroll-166"
agent_name = 'data_fix_512_1_20000000_292clt4b_lunar-kumquat-192'
surrogate_name = "data_fix_512_1_2000000_33y4mv4t_golden-dumpling-194"
pt_surrogate_name = "data_fix_512_1_2000000_14uukmw7_dancing-fish-193"
env_surrogate_act = torch.tanh
env_surrogate = SimpleSurrogate(target_size=surrogate_actions[0].shape[0], hidden_size=512, num_hidden=surrogate_num_hidden, activation = env_surrogate_act, lr=surrogate_lr, last_layer_activation = last_layer_act_surrogate)
#env_surrogate.model.load_state_dict(torch.load(pt_surrogate_name, map_location=device))
env_surrogate.model.load_state_dict(torch.load('/home/bethke52/laser_data/historic_data/surrogates/'+ pt_surrogate_name, map_location=device))

agent_lr = 1e-6
surrogate_lr = 1e-5
continuous = False

if continuous:
    agent = SimpleAgent(input_size=states[0].shape[0], target_size=actions[0].shape[0], hidden_size=512, num_hidden=agent_num_hidden, activation = torch.sigmoid, lr=agent_lr, last_layer_activation = last_layer_act_agent, norm = norm_feature,  clip_norm = clip_norm, minmax_norm=minmax_norm)
    surrogate = SimpleSurrogate(target_size=surrogate_actions[0].shape[0], hidden_size=512, num_hidden=surrogate_num_hidden_, activation = torch.tanh, lr=surrogate_lr, last_layer_activation = last_layer_act_surrogate)
    agent.agent.load_state_dict(torch.load("/home/bethke52/laser_data/historic_data/models/" +agent_name))
    surrogate.model.load_state_dict(torch.load("/home/bethke52/laser_data/historic_data/surrogates/"+surrogate_name))


data = {}

improvement_track = []
for idx in range(pre_start_states.shape[0], states.shape[0]):
    target_data = {}
    #idx = 52
    start = states[idx,:2048].unsqueeze(0).float().to(device)
    goal = states[idx,2048:].unsqueeze(0).float().to(device)
    action = actions[idx].unsqueeze(0).float().to(device)
    
    if not continuous:
        agent = SimpleAgent(input_size=states[0].shape[0], target_size=actions[0].shape[0], hidden_size=512, num_hidden=agent_num_hidden, activation = torch.sigmoid, lr=agent_lr, last_layer_activation = last_layer_act_agent, norm = norm_feature,  clip_norm = clip_norm, minmax_norm=minmax_norm)
        surrogate = SimpleSurrogate(target_size=surrogate_actions[0].shape[0], hidden_size=512, num_hidden=surrogate_num_hidden_, activation = torch.tanh, lr=surrogate_lr, last_layer_activation = last_layer_act_surrogate)
        agent.agent.load_state_dict(torch.load("/home/bethke52/laser_data/historic_data/models/" +agent_name))
        surrogate.model.load_state_dict(torch.load("/home/bethke52/laser_data/historic_data/surrogates/"+surrogate_name))

    env = Environment(targets=goal, state=start, surrogate=env_surrogate)
    tuned_state_base = agent.inference(1,env).unsqueeze(0)

    agent.experiences.vip_state_memory = pre_start_states.float().to(device) 
    agent.experiences.vip_action_memory = pre_actions.float().to(device)
    agent.experiences.vip_next_state_memory = pre_next_states.float().to(device)
    agent.experiences.vip_org_target_memory = pre_next_states.float().to(device) 

    train_params = {'batch_size':64, 'update_every':1}
    loss = []

    env = Environment(targets=goal, state=start, surrogate=env_surrogate)

    loss = loss + agent.explore(env, num_episodes= 4000, episode_len=1, train_params=train_params, surrogate=surrogate, prior=False, ssim=False, smooth_state=False)


    tuned_state_optim = agent.inference(1,env).unsqueeze(0)

    improvement_track.append((agent.loss_f(goal,tuned_state_base).item(),agent.loss_f(goal,tuned_state_optim).item()))
    agent_out = agent.agent(states[idx].unsqueeze(0).float().to(device)).cpu().detach().numpy()[0]
    
    target_data["loss"] = loss
    target_data["tuned"] = tuned_state_optim
    target_data["base"] = tuned_state_base
    target_data["agent_out"] = agent_out
    data[idx] = target_data
data["improvement_track"] = improvement_track

np.save("eval_single_train_data_{}_{}_{}_{}_{}".format(continuous,last_layer_act_agent, agent_name.split('_')[5], surrogate_name.split('_')[5], pt_surrogate_name.split('_')[5]), data)