import wandb
import sys
sys.path.append('..')
sys.path.append('../rl_tests/additive_rework/')
import numpy as np
import torch
import matplotlib.pyplot as plt
from mlp import *
from tqdm import tqdm
import itertools
from utils import *
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
from environment import *
from simple_agent import *
from simple_surrogate import *

parser = ArgumentParser()
# Model
parser.add_argument("--continue_train", type=str, default= None)
parser.add_argument("--savedir", type = str, default = None)
parser.add_argument("--num_hidden", type = int, default = 1)
parser.add_argument("--hidden_size", type = int, default = 64)

# Dataset
# Batch size 6o sampled from buffer 
parser.add_argument("--batch_size", type = int, default = 48)
# Update every *update_every* episode
parser.add_argument("--update_every", type = int, default = 1)

# Optimization
parser.add_argument("--lr", type = float, default = 1e-4)
parser.add_argument("--n_episodes", type = int, default = 1000)
parser.add_argument("--episode_len", type = int, default = 1)

# Log
parser.add_argument("--log_writer", type=int, default=0)
parser.add_argument("--log_every", type=int, default=100)

args = parser.parse_args()
if args.log_writer:
    wandb.init(project='Laser-Agents', entity='aipp')
    wandb.run.save()
    wandb.config.update(args)
else:
    log_writer = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)

# Size of the targets below
target_shape = 100
# Manually created target signals
pointy_target = norm(torch.cat([torch.zeros(25),torch.arange(25),torch.flip(torch.arange(25), (0,)), torch.zeros(25)]), norm_factor=2)
step_target = norm(torch.cat([torch.zeros(40),torch.ones(20), torch.zeros(40)]), norm_factor=4)
step_target_small = norm(torch.cat([torch.zeros(40),torch.ones(20), torch.zeros(40)]), norm_factor=3)
ramp_target = norm(torch.cat([torch.zeros(40),torch.arange(20), torch.zeros(40)]), norm_factor=2)
ramp_target_rev = torch.flip(norm(torch.cat([torch.zeros(40),torch.arange(20), torch.zeros(40)]), norm_factor=2), (0,))
gauss_targets= [gauss_range(100, mu=np.random.uniform(-2,2), sigma=np.random.uniform(0.5,3), norm_factor=2) for i in range(5)]

targets = torch.stack([pointy_target,step_target, step_target_small, ramp_target, ramp_target_rev, *gauss_targets]).to(device)

# Initalize environment with targets
env = Environment(targets)

# Initalize agent
agent = SimpleAgent(input_size=target_shape*2, target_size=target_shape, hidden_size=args.hidden_size, num_hidden=args.num_hidden, activation = torch.tanh, lr=args.lr)
# Initalize Surrogate
surrogate = SimpleSurrogate(target_size=target_shape, lr=args.lr,num_hidden=args.num_hidden)
print(agent.agent)

model_name = '{}_{}_{}'.format(args.hidden_size, args.num_hidden, args.n_episodes)
# Initialize W&B log writer
if args.log_writer:
        log_writer_type = 'wb'
        log_writer = wandb
        wandb.watch(agent.agent, log="all")
        log_writer=(log_writer_type,log_writer)
        model_name += '_'+log_writer[1].run.id +'_'+ log_writer[1].run.name

train_params = {'batch_size':args.batch_size, 'update_every':args.update_every}

# Start exploration based training
agent.explore(env, num_episodes= args.n_episodes, episode_len=args.episode_len, train_params=train_params, surrogate=surrogate, prior=False, ssim=False, smooth_state=False, log_writer = log_writer, log_every=args.log_every, model_name= args.savedir+model_name)
if args.savedir != None:
    torch.save(agent.agent.state_dict(), args.savedir+model_name)