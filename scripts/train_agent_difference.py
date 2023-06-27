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

#dataset
parser.add_argument("--batch_size", type = int, default = 72)

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

states, actions = get_data(difference_model=True)

train_dataset = torch.utils.data.TensorDataset(states.float().to(device), actions.float().to(device))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

agent = SimpleAgent(input_size=states[0].shape[0], target_size=actions[0].shape[0], hidden_size=args.hidden_size, num_hidden=args.num_hidden, activation = torch.tanh, lr=args.lr, last_layer_activation = True if args.last_layer_activation else False)
print(agent.agent)
model_name = '{}_{}_{}'.format(args.hidden_size, args.num_hidden, args.n_epochs)
if args.log_writer:
        log_writer_type = 'wb'
        log_writer = wandb
        wandb.watch(agent.agent, log="all")
        log_writer=(log_writer_type,log_writer)
        model_name += '_'+log_writer[1].run.id +'_'+ log_writer[1].run.name
        
agent.pretrain(train_loader, num_epochs=args.n_epochs, log_writer=log_writer)
torch.save(agent.agent.state_dict(), args.savedir+model_name)