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

plt.style.use('dark_background')

# Model selection 

#pt_surrogate_name = '512_2_20000000_1e7vg3o2_eternal-glade-99' # base 2 512 layer
#pt_surrogate_name = '512_2_20000000_3qwdlj58_earnest-vortex-124' # smooth norm 2 512 layer
#pt_surrogate_name =  '512_2_20000000_1luocgbd_jolly-silence-130'# smooth norm 2 512 layer earnest-vortex continue
#pt_surrogate_name = '512_2_20000000_cw0oap87_bright-cosmos-141' # jolly silence continue 175 samples
#pt_surrogate_name = '512_2_20000000_tkfo6rnj_celestial-fog-142' # bright cosmos continue
#pt_surrogate_name = '512_1_20000000_1dntv83z_stellar-brook-148' # 1 512 layer
#pt_surrogate_name = '512_1_20000000_34gaeeih_dauntless-shadow-151' # stellar brook continue
pt_surrogate_name = '512_1_20000000_3w0kc0cs_upbeat-meadow-153' # ds continue


surrogate_num_hidden = 1
agent_num_hidden = 1
# Optimization learning rates
agent_lr = 1e-6
surrogate_lr = 1e-6

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
last_layer_act_agent=False
last_layer_act_surrogate=True

x_vals_setting = np.load('../setting_x_values.npy')
x_vals_spectrum = np.load('../spectrum_x_values.npy')


states, actions = get_data(smooth_states=smooth_states, norm_states = norm_states)
start_states = states[:,:int(states.shape[1]/2)]
next_states = states[:,int(states.shape[1]/2):]
surrogate_states = torch.cat([start_states, actions], dim=1)
surrogate_actions = next_states

agent = SimpleAgent(input_size=states[0].shape[0], target_size=actions[0].shape[0], hidden_size=512, num_hidden=agent_num_hidden, activation = torch.sigmoid, lr=agent_lr, last_layer_activation = last_layer_act_agent, norm = norm_feature,  clip_norm = clip_norm, minmax_norm=minmax_norm)
surrogate = SimpleSurrogate(target_size=surrogate_actions[0].shape[0], hidden_size=512, num_hidden=surrogate_num_hidden, activation = torch.sigmoid, lr=surrogate_lr, last_layer_activation = last_layer_act_surrogate)


env_surrogate = SimpleSurrogate(target_size=surrogate_actions[0].shape[0], hidden_size=512, num_hidden=surrogate_num_hidden, activation = torch.sigmoid, lr=surrogate_lr, last_layer_activation = last_layer_act_surrogate)
env_surrogate.model.load_state_dict(torch.load('/home/bethke52/laser_data/historic_data/surrogates/'+ pt_surrogate_name, map_location=device))


max_sample_day_org_idx = np.argmax(np.unique(states[:,:2000], axis=0, return_counts=True)[1])
max_sample_day_start = np.unique(states[:,:2000], axis=0)[max_sample_day_org_idx]
max_sample_day_idx_list = np.where(np.isclose(states[:,:2000], max_sample_day_start).sum(axis=1)==2000)[0]
start = states[max_sample_day_idx_list[0],:2048].unsqueeze(0).float().to(device)
goal = states[max_sample_day_idx_list,2048:].float().to(device)
action = actions[max_sample_day_idx_list].unsqueeze(0).float().to(device)

idx = max_sample_day_idx_list
pretrain_goal = norm(gauss_range(2048, bound=7)).unsqueeze(0).float().to(device)
loss_track= []
for i in tqdm(range(10000)):
    agent.optim.zero_grad()
    out = agent.agent(states[idx,:].float().to(device))
    loss = agent.loss_f(out,pretrain_goal.repeat(idx.shape[0],1))
    loss_track.append(loss.item())
    loss.backward()
    agent.optim.step()
    
train_params = {'batch_size':1024, 'update_every':1}
num_episodes = 1000000
#num_episodes = 10
loss = []

env = Environment(targets=goal, state=start, surrogate=env_surrogate)
loss = loss + agent.explore(env, num_episodes= num_episodes , episode_len=1, train_params=train_params, surrogate=surrogate, prior=False, ssim=False, smooth_state=False)

np.save("env_surrogate_test_multi_target_{}_{}_{}_{}_{}".format(num_episodes, train_params["batch_size"], agent_lr, surrogate_lr, last_layer_act_agent),{"loss": loss, "agent":agent.agent.state_dict(),"surrogate":surrogate.model.state_dict(), "train_params":train_params, "lr":(agent_lr, surrogate_lr)})