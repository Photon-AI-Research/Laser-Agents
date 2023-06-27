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

from simple_agent import *

parser = ArgumentParser()
#model
parser.add_argument("--continue_train", type=str, default= None)
parser.add_argument("--savedir", type = str, default = None)
parser.add_argument("--num_hidden", type = int, default = 1)
parser.add_argument("--hidden_size", type = int, default = 64)
parser.add_argument("--last_layer_activation", type = int, default = 1)
parser.add_argument("--activation", type = str, default = "sigmoid")
parser.add_argument("--norm", type = int, default = 0)
parser.add_argument("--clip_norm", type = int, default = 0)
parser.add_argument("--minmax_norm", type = int, default = 1)
parser.add_argument("--norm_power", type = int, default = 0)
#dataset
parser.add_argument("--batch_size", type = int, default = 46)
parser.add_argument("--train_with_loop_data", type = int, default = 0)
parser.add_argument("--smooth_states", type = int, default = 0)
parser.add_argument("--norm_states", type = int, default = 0)
parser.add_argument("--include_bad_examples", type = int, default = 0)
parser.add_argument("--num_bad", type = int, default = 2000)
parser.add_argument("--val_train", type = int, default = 0)
parser.add_argument("--pwl_train", type = int, default = 0)

#optimization
parser.add_argument("--lr", type = float, default = 1e-4)
parser.add_argument("--n_epochs", type = int, default = 1000)

#log
parser.add_argument("--log_writer", type=int, default=0)

args = parser.parse_args()
if args.log_writer:
    wandb.init(project='Laser-Agents', entity='aipp')
    wandb.run.save()
    wandb.config.update(args)
else:
    log_writer = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234)


states, actions = get_data(train_with_live_data= True if args.train_with_loop_data else False,smooth_states = True if args.smooth_states else False,norm_states = True if args.norm_states else False, norm_power = True if args.norm_power else False)
print('Training with {} samples'.format(len(states)))

start_state = states[:,:int(states.shape[1]/2)]
next_state = states[:,int(states.shape[1]/2):]
surrogate_states = torch.cat([start_state, actions], dim=1)
surrogate_actions = next_state

if args.include_bad_examples:
    bad_waves = create_bad_examples(actions[:,1:].numpy(), num_mod=int(args.num_bad//2), num_noise=int(args.num_bad//2))
    bad_settings = torch.tensor(np.concatenate((np.random.uniform(0,0.5,size=(bad_waves.shape[0],1)),bad_waves), axis=1))
    bad_surrogate_actions = -1* torch.ones(bad_settings.shape)
    bad_surrogate_states = torch.cat([start_state[np.random.randint(0,surrogate_states.shape[0],size=bad_settings.shape[0])], bad_settings], dim=1)

    surrogate_states = torch.cat([surrogate_states, bad_surrogate_states],dim=0)
    surrogate_actions = torch.cat([surrogate_actions, bad_surrogate_actions],dim=0)

if args.pwl_train:
    base_states, base_actions = get_data(train_with_live_data= True, smooth_states = True if args.smooth_states else False,norm_states = True if args.norm_states else False, norm_power = True if args.norm_power else False)
    surrogate_states, xn, xp = get_pwl_surrogate_states(surrogate_states, base_waves=base_actions[:,1:])
    
val_loader = None
if args.val_train:
    np.random.seed(1234)
    val_size = 5 
    val_idx = np.random.choice(np.arange(46),val_size, replace=False)
    train_idx = np.setxor1d(np.arange(surrogate_states.shape[0]), val_idx)
    train_dataset = torch.utils.data.TensorDataset(surrogate_states[train_idx].float().to(device), surrogate_actions[train_idx].float().to(device))
    val_dataset = torch.utils.data.TensorDataset(surrogate_states[val_idx].float().to(device), surrogate_actions[val_idx].float().to(device))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size-val_size, shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_size, shuffle=False, drop_last=False)
else:
    train_dataset = torch.utils.data.TensorDataset(surrogate_states.float().to(device), surrogate_actions.float().to(device))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)



if args.activation == 'tanh':
    activation = torch.tanh
else:
    activation = torch.sigmoid


agent = SimpleAgent(input_size=surrogate_states[0].shape[0], target_size=surrogate_actions[0].shape[0], hidden_size=args.hidden_size, num_hidden=args.num_hidden, activation = activation, lr=args.lr, last_layer_activation = True if args.last_layer_activation else False, norm = True if args.norm else False,  clip_norm = True if args.clip_norm else False,minmax_norm = True if args.minmax_norm else False )
print(agent.agent)
if args.continue_train != None:
    agent.agent.load_state_dict(torch.load('/home/bethke52/laser_data/historic_data/surrogates/'+ args.continue_train, map_location=device))
    print('State loaded')
model_name = 'loop_data_fix_{}_{}_{}'.format(args.hidden_size, args.num_hidden, args.n_epochs)
if args.log_writer:
        log_writer_type = 'wb'
        log_writer = wandb
        wandb.watch(agent.agent, log="all")
        log_writer=(log_writer_type,log_writer)
        model_name += '_'+log_writer[1].run.id +'_'+ log_writer[1].run.name
        
agent.pretrain(train_loader, num_epochs=args.n_epochs, log_writer=log_writer, val_loader = val_loader)
torch.save(agent.agent.state_dict(), args.savedir+model_name)