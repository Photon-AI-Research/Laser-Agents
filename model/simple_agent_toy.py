import numpy as np
import torch
from mlp_conv import *
from mlp import *
from buffer import *
from tqdm import tqdm
from ssim import SSIM
import sys
sys.path.append('../..')
from utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# A simple agent that predicts the inverese of a surrogate
class SimpleAgent():
    def __init__(self, target_size, input_size = None, hidden_size=None, num_hidden=1, activation=torch.tanh, lr=1e-4, loss_f=torch.nn.MSELoss(), conv=False, kernel_size=7, last_layer_activation = True, norm=False, clip_norm=False, minmax_norm=True, buffer_size= 50000):
        hidden_size = target_size if hidden_size == None else hidden_size
        input_size = target_size*2 if input_size == None else input_size
        # Model
        if conv: 
            # Model with additional convolution layer (experimental)
            self.agent = MLP_Conv(input_size, target_size, hidden_size, num_hidden, activation = activation, kernel_size=kernel_size).to(device)
        else:
            self.agent = MLP(input_size, target_size, hidden_size, num_hidden, activation = activation, last_layer_activation=last_layer_activation, norm=norm, clip_norm=clip_norm, minmax_norm=minmax_norm).to(device)
        # Experience buffer
        self.experiences = Buffer(buffer_size)
        
        # Optimization
        self.optim = torch.optim.Adam(self.agent.parameters(),lr=lr)
        self.loss_f = loss_f
    
    # Offline training of the agent using a data loader
    def pretrain(self, train_loader, num_epochs=1000, log_writer=None, num_total_plots = 1, num_plots = 1, plot = False, downsampling_rate=None, log_every=1000, model_name = None):
        # Loss track
        loss_evol = []
        # Pretrain
        pbar = tqdm(range(num_epochs))
        # Epochs
        for epoch in pbar:
            loss_epoch = []
            # Iterations
            for state, action in train_loader:
                self.optim.zero_grad()
                predicted_action = self.agent(state)
                loss = self.loss_f(predicted_action, action)
                loss_epoch.append(loss.item())
                loss.backward()
                self.optim.step()
            # Logging functionality
            train_loss = np.mean(loss_epoch)
            loss_evol.append(train_loss)
            pbar.set_postfix({'Train loss': train_loss})
            if log_writer != None and epoch % log_every == 0:
                if log_writer[0] == 'wb':
                    log_writer[1].log({'Train loss': train_loss, 'Epoch': epoch})
                if epoch % (log_every*10) and epoch != 0 and model_name != None:
                    torch.save(self.agent.state_dict(), model_name)
            # Plotting functionality
            if plot and epoch % (num_epochs//num_total_plots) == 0 and epoch != 0:
                states, actions =train_loader.dataset[:]
                if epoch == (num_epochs//num_total_plots):
                    with torch.no_grad():
                        val_losses = torch.mean(torch.nn.MSELoss(reduction='none')(self.agent(states.float().to(device)),actions.float().to(device)),dim=1)
                    val_losses_idx = torch.argsort(val_losses).flip(dims=(0,))
                # Log evaluation plot (in this case for the Laser-Agent project)
                action_eval_plot(self.agent, num_plots, states, actions, log_writer=log_writer, index = val_losses_idx, downsampling_rate = downsampling_rate, epoch=epoch)
            
        return(np.array(loss_evol))
    
    # Exploration based training of the agent
    def explore(self, env, smooth_state=False, num_episodes=1000, episode_len=50, train_params=None, surrogate=None, prior=True, ssim=False, log_writer=None, log_every=100, model_name=None):
        # Smooth state returned from env for next state
        if smooth_state:
            kernel = torch.autograd.Variable(torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]])).to(device)
        loss_track = []
        # Explore *num_episodes* times on random targets of the env
        pbar = tqdm(range(num_episodes))
        for episode in pbar:
            # Perform *episode_len* steps in one episode
            for step in range(episode_len):
                # Smoothing of the state returned from the environment (experimental)
                if smooth_state:
                    env.state = torch.nn.functional.conv1d(env.state.unsqueeze(0),kernel, padding=3)[0]
                # Set current start state
                start_state = env.state
                # Infer agent
                with torch.no_grad():
                    predicted_action = self.agent(torch.cat([env.state,env.target]).unsqueeze(0))
                # Make environment step
                env.step(predicted_action)
                # Get next state
                next_state = env.state
                # Add gathered data to experiences and perform optimization step (if possible)
                if surrogate == None:
                    loss = self.step(start_state, predicted_action[0], next_state, env.target, train_params, prior)
                else:
                    loss = self.surrogate_step(start_state, predicted_action[0], next_state, env.target, train_params, surrogate, prior, ssim)
                # Logging functionality
                loss_track.append(loss) if loss != None else None
                pbar.set_postfix({'Explore loss': loss_track[-1]}) if len(loss_track) != 0 else None
                if log_writer != None and loss != None and episode % log_every ==0:
                    log_writer[1].log({"Loss": loss, "Episode": episode})
                    # Save current model state
                    if model_name != None:
                        torch.save(self.agent.state_dict(), model_name)
                        torch.save(surrogate.model.state_dict(), model_name+'_surrogate')
            # Resent environment after episode
            env.reset()
        return loss_track
    
    # Perform a agent step without critic on exploration data
    def step(self, start_state, predicted_action, next_state, org_target, train_params, prior):  
        self.experiences.add(start_state, predicted_action, org_target, next_state)
        # Learn from experiences
        if train_params != None and self.experiences.state_memory.shape[0] % train_params['update_every'] == 0 and self.experiences.state_memory.shape[0]>train_params['batch_size'] :
            self.optim.zero_grad()
            states, actions, targets, _ = self.experiences.sample(train_params['batch_size'], prior=prior)
            predicted_action = self.agent(torch.cat([states, targets],dim=1))
            loss = self.loss_f(predicted_action, actions)
            loss.backward()
            self.optim.step()
            return loss.item()
    
    # Perform an agent optimization step in Actor-Critic fashion using a surrogate
    def surrogate_step(self, start_state, predicted_action, next_state, org_target, train_params, surrogate, prior, ssim):  
        self.experiences.add(start_state, predicted_action, next_state, org_target)
        # Learn from experiences
        if train_params != None and self.experiences.state_memory.shape[0] % train_params['update_every'] == 0 and self.experiences.state_memory.shape[0]>=train_params['batch_size'] :
            return self.surrogate_learn_from_exp(surrogate, train_params, prior, ssim)
    
    # Actor (agent) - Critic (surrogate), optimization step
    def surrogate_learn_from_exp(self, surrogate, train_params, prior, ssim=False, rollout=1, surrogate_step=True):
        # Sample a batch from the buffer
        states, actions, next_states, org_targets = self.experiences.sample(train_params['batch_size'], prior=prior)
        # Critic optimization step
        if surrogate_step:
            surrogate.optim.zero_grad()
            predicted_next_states = surrogate.model(torch.cat([states, actions],dim=1))
            surrogate_loss = surrogate.loss_f(predicted_next_states, next_states)
            surrogate_loss.backward()
            surrogate.optim.step()
        
        # Actor optimization step, with rollout functionality (experimental)
        for i in range(rollout):
            self.optim.zero_grad()
            predicted_actions = self.agent(torch.cat([states, org_targets],dim=1))
            predicted_next_states = surrogate.model(torch.cat([states,predicted_actions],dim=1))
            if ssim:
                # Shape-based objective function
                agent_loss = -SSIM()(predicted_next_states.unsqueeze(1), org_targets.unsqueeze(1))
            else:
                agent_loss = self.loss_f(predicted_next_states, org_targets)
            agent_loss.backward()
            self.optim.step()
        return agent_loss.item()
    
    # Train a surrogate with the buffer of this agent
    def surrogate_train_from_exp(self, surrogate, train_params, num_iterations=1, prior=False, ssim=False):
        pbar = tqdm(range(num_iterations))
        loss_track = []
        for iteration in pbar:
            loss = self.surrogate_learn_from_exp(surrogate, train_params, prior, ssim=False)
            loss_track.append(loss)
            pbar.set_postfix({'Exp train loss': loss})
        return loss_track
    
    # Infer model given an environment and a episode length
    def inference(self, episode_len, env):
        for step in range(episode_len):
            with torch.no_grad():
                predicted_action = self.agent(torch.cat([env.state,env.target]).unsqueeze(0))
            env.step(predicted_action)
        return env.state
        
        