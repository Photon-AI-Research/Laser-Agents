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
agent_lr = 1e-6
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
last_layer_act_agent=True
last_layer_act_surrogate=True

x_vals_setting = np.load('../setting_x_values.npy')
x_vals_spectrum = np.load('../spectrum_x_values.npy')

norm_power = False

states, base_actions = get_data(smooth_states=smooth_states, norm_states = norm_states, norm_power=norm_power)
states = states[:175]
base_actions = base_actions[:175]
start_states = states[:,:int(states.shape[1]/2)]
next_states = states[:,int(states.shape[1]/2):]
base_surrogate_states = torch.cat([start_states, base_actions], dim=1)
surrogate_actions = next_states

actions, xn, xp = get_pwl_actions(base_actions.float().to(device))
surrogate_states, xn, xp = get_pwl_surrogate_states(base_surrogate_states)

pre_states, base_pre_actions = get_data(smooth_states=smooth_states, norm_states = norm_states, train_with_live_data=False, norm_power=norm_power)
pre_start_states = pre_states[:,:int(pre_states.shape[1]/2)]
pre_next_states = pre_states[:,int(pre_states.shape[1]/2):]
base_pre_surrogate_states = torch.cat([pre_start_states, base_pre_actions], dim=1)
pre_surrogate_actions = pre_next_states

pre_actions, xn, xp = get_pwl_actions(base_pre_actions.float().to(device), base_waves=base_actions[:,1:])
pre_surrogate_states, xn, xp = get_pwl_surrogate_states(base_pre_surrogate_states, base_waves=base_actions[:,1:])

last_layer_act_agent=True
agent_num_hidden = 1
surrogate_num_hidden_ = 1
agent = SimpleAgent(input_size=states[0].shape[0], target_size=actions[0].shape[0], hidden_size=512, num_hidden=agent_num_hidden, activation = torch.sigmoid, lr=agent_lr, last_layer_activation = last_layer_act_agent, norm = norm_feature,  clip_norm = clip_norm, minmax_norm=minmax_norm,xn=xn, xp=xp)
surrogate = SimpleSurrogate(target_size=surrogate_actions[0].shape[0], hidden_size=512, num_hidden=surrogate_num_hidden_, activation = torch.tanh, lr=surrogate_lr, last_layer_activation = last_layer_act_surrogate, input_size=surrogate_states.shape[1])

agent_name = "pwl_noloopfix_data_fix_512_1_20000000_3s3fnhtx_noble-cosmos-257"
# with bad
#surrogate_name = "loop_data_fix_512_1_2000000_2bdp8ofe_atomic-butterfly-250"

# no bad -> works
#surrogate_name = "loop_data_fix_512_1_10000000_2394w3v7_gentle-feather-265"
#with bad continue gf -> works
surrogate_name = "loop_data_fix_512_1_2000000_bvjnqe2b_glorious-energy-266"

# with bad 
#pt_surrogate_name = "loop_data_fix_512_1_2000000_2i1e9kyl_mild-violet-253"
# with bad and lab
pt_surrogate_name = 'loop_data_fix_512_1_10000000_31x86z8h_beloved-sweetheart-263'
# no bad
#pt_surrogate_name = "data_fix_512_1_2000000_122vzfqv_giddy-armadillo-223"

# norm power with bad
#pt_surrogate_name ='data_fix_512_1_2000000_2o3nqoa2_worldly-silence-229'

agent.agent.load_state_dict(torch.load("/home/bethke52/laser_data/historic_data/models/" +agent_name))
surrogate.model.load_state_dict(torch.load("/home/bethke52/laser_data/historic_data/surrogates/"+surrogate_name))

env_surrogate_act = torch.tanh
env_surrogate = SimpleSurrogate(target_size=surrogate_actions[0].shape[0], input_size=surrogate_states.shape[1], hidden_size=512, num_hidden=surrogate_num_hidden, activation = env_surrogate_act, lr=surrogate_lr, last_layer_activation = last_layer_act_surrogate)
env_surrogate.model.load_state_dict(torch.load('/home/bethke52/laser_data/historic_data/surrogates/'+ pt_surrogate_name, map_location=device))


agent_lr = 1e-6
surrogate_lr = 1e-5
continuous = False
savety_check = False
data = {}
improvement_track = []
for idx in range(pre_start_states.shape[0], states.shape[0]):
    target_data = {}
    start = states[idx,:2048].unsqueeze(0).float().to(device)
    goal = states[idx,2048:].unsqueeze(0).float().to(device)
    action = base_actions[idx].unsqueeze(0).float().to(device)
    
    agent = SimpleAgent(input_size=states[0].shape[0], target_size=actions[0].shape[0], hidden_size=512, num_hidden=agent_num_hidden, activation = torch.sigmoid, lr=agent_lr, last_layer_activation = last_layer_act_agent, norm = norm_feature,  clip_norm = clip_norm, minmax_norm=minmax_norm,xn=xn, xp=xp)
    surrogate = SimpleSurrogate(target_size=surrogate_actions[0].shape[0], hidden_size=512, num_hidden=surrogate_num_hidden_, activation = torch.tanh, lr=surrogate_lr, last_layer_activation = last_layer_act_surrogate, input_size=surrogate_states.shape[1])
    agent.agent.load_state_dict(torch.load("/home/bethke52/laser_data/historic_data/models/" +agent_name))
    surrogate.model.load_state_dict(torch.load("/home/bethke52/laser_data/historic_data/surrogates/"+surrogate_name))
    
    env = Environment(targets=goal, state=start, surrogate=env_surrogate, savety_check=savety_check)
    tuned_state_base = agent.inference(1,env).unsqueeze(0)
    agent.agent.pwl = True
    agent_out_base_full = agent.agent(states[idx].unsqueeze(0).float().to(device)).cpu().detach().numpy()[0]
    agent.agent.pwl = False
    agent_out_base = agent.agent(states[idx].unsqueeze(0).float().to(device)).cpu().detach().numpy()[0]
    
    agent.experiences.vip_state_memory = pre_start_states.float().to(device) 
    agent.experiences.vip_action_memory = pre_actions.float().to(device)
    agent.experiences.vip_next_state_memory = pre_next_states.float().to(device)
    agent.experiences.vip_org_target_memory = pre_next_states.float().to(device) 

    train_params = {'batch_size':64, 'update_every':1}
    loss = []

    env = Environment(targets=goal, state=start, surrogate=env_surrogate, savety_check=savety_check)
    
    num_episodes = 10000
    loss = loss + agent.explore(env, num_episodes= num_episodes, episode_len=1, train_params=train_params, surrogate=surrogate, prior=False, ssim=False, smooth_state=False)
    #plt.plot(loss[500:])
    
    tuned_state_optim = agent.inference(1,env).unsqueeze(0)
    agent.agent.pwl = True
    agent_out_tuned_full = agent.agent(states[idx].unsqueeze(0).float().to(device)).cpu().detach().numpy()[0]
    
    
    improvement_track.append((agent.loss_f(goal,tuned_state_base).item(),agent.loss_f(goal,tuned_state_optim).item()))
    target_data["loss"] = loss
    target_data["tuned"] = tuned_state_optim
    target_data["base"] = tuned_state_base
    target_data["agent_out"] = agent_out_base
    target_data["agent_out_tuned_full"] = agent_out_tuned_full
    target_data["agent_out_base_full"] = agent_out_base_full
    data[idx] = target_data
data["improvement_track"] = improvement_track

np.save("eval_single_train_data_{}_{}_{}_{}_{}_{}".format(continuous,last_layer_act_agent, agent_name.split('_')[6], surrogate_name.split('_')[6], pt_surrogate_name.split('_')[6], num_episodes), data)