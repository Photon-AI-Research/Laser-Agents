import numpy as np
from torch.nn.functional import mse_loss
import torch
import torch.nn.functional as F
import itertools
from os import listdir
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import resample

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Utility functions for Mazzler interaction and corresponding models

# Load data from numpy files
def get_data(train_with_live_data=True, difference_model=False, smooth_states=False, kernel_size = 25, norm_states=False, train_with_hole_data=False, norm_power=False, norm_power_max=0.05):
    if difference_model:
        states = torch.tensor([])
        actions = torch.tensor([])
    else:
        power_set = np.load('/home/bethke52/laser_data/historic_data/historic_power_set.npy')#[:1]
        setting_set = np.load('/home/bethke52/laser_data/historic_data/historic_setting_set.npy')#[:1]
        spectrum_set = np.load('/home/bethke52/laser_data/historic_data/historic_spectra_set.npy')#[:1]

        states = torch.tensor(np.array([np.concatenate([spectrum_set[i,0,:,1]/3500,spectrum_set[i,1,:,1]/2000]) for i in range(spectrum_set.shape[0])]))
        actions = torch.tensor(np.array([np.concatenate([np.array([power_set[i]]),setting_set[i,:,1]]) for i in range(setting_set.shape[0])]))
    if train_with_live_data:
        live_sets = [["live_set_22_09_12","live_2022_09_12/spectra/2022.09.12_13.37.17_start2.txt"],
                         ["live_set_22_09_26","live_2022_09_26/new_set/26/2022.09.26_13.25.18_start.txt"],
                         ["live_set_22_10_14","2022_10_14_spectrumAI/loop_set/spectra/2022.10.14_11.33.58_loop_start.txt"],
                         ["model_set_22_09_26","live_2022_09_26/new_set/26/2022.09.26_13.25.18_start.txt"],
                         ["model_set_22_10_14","2022_10_14_spectrumAI/model_set/spectra/2022.10.14_10.41.59_start.txt"],
                         ["model_set_22_10_20","live_2022_10_20/spectra/model/2022.10.20_14.34.45_start2.txt"],
                         ["model_set_22_11_08","20221108FriedrichBethke/spectra/08/2022.11.08_14.26.38_ohne_mazzler.txt"],
                         ["model_set_22_11_10","2022_11_10_Friedrich_Daten/spectra/2022.11.10_10.12.04_start.txt"],
                         ["model_set_23_02_14", "live_2023_02_14/spectra/start_2_2023.02.14_10.57.48.txt"],
                         ["model_set_23_02_15", 'live_2023_02_15/spectra/start_1_2023.02.15_08.58.08.txt']
                        ]
        if train_with_hole_data:
            live_sets.append(["hole_live_set_22_10_20","live_2022_10_20/spectra/loop/2022.10.20_13.53.12_start.txt"])
        for live_set in live_sets:
            # Load data
            power_loop = np.load('/home/bethke52/laser_data/live_data/power_{}.npy'.format(live_set[0]))
            setting_loop= np.load('/home/bethke52/laser_data/live_data/setting_base_{}.npy'.format(live_set[0]))
            spectrum_loop = np.load('/home/bethke52/laser_data/live_data/spectrum_{}.npy'.format(live_set[0]))
            spectrum_start = np.loadtxt('/home/bethke52/laser_data/{}'.format(live_set[1]))
            # norm step
            setting_loop= norm_np(setting_loop, axis=1)

            # Get state/action pairs in historic data fashion
            if difference_model:
                loop_states, loop_actions = get_state_acion_from_loop_set(power_loop, setting_loop, spectrum_loop)
            else:
                loop_states, loop_actions = get_historic_states_actions(spectrum_start,spectrum_loop,power_loop,setting_loop)
            states = torch.cat([states, loop_states])
            actions = torch.cat([actions,loop_actions])            
        #if smooth_states:
        #    states = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)(states)
        #if norm_states:
        #    states = torch.cat([norm_tensor(states[:,:2048], dim=1),norm_tensor(states[:,2048:], dim=1)], dim=1)
    if smooth_states:
        states = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)(states)
    if norm_states:
        states = torch.cat([norm_tensor(states[:,:2048], dim=1),norm_tensor(states[:,2048:], dim=1)], dim=1)
    if norm_power:
        actions[:,0] = actions[:,0]/norm_power_max
    return states, actions

# Write setting files for mazzler
def write_setting(power, wave_values, norm, file_name, base_setting='./live_2022_09_12/20220912FriedrichBethke/base_config',base_path='generated_settings/'):
    with open(base_setting) as f:
        lines = np.array(f.readlines())
    lines[2136] = 'Norm = +{:.6f}E-3\n'.format(norm*1000)
    lines[17] = 'power={:.6f}\n'.format(power)
    lines[22:2069] = np.array([lines[22:2069][i].split('\t')[0]+'\t'+'{:.6f}'.format(wave_values[i])+'\n' for i in range(len(lines[22:2069]))])
    with open(base_path+file_name, 'w') as f:
        f.writelines(lines)
    return

# Read setting files from mazzler
def read_setting(path, return_lines = False):
    with open(path) as f:
        lines = np.array(f.readlines())
    if return_lines:
        return lines
    else:
        power = float(lines[17][6:-1])
        setting = np.array([i[:-1].split('\t') for i in lines[22:2069]]).astype(float)
        norm = float(lines[2136][7:])
        return setting, power, norm

# A funicton to test a model with a loader and mse loss
def test(loader, model):
    model.eval()
    loss = 0
    for x,y in loader:
        with torch.no_grad():
            loss += mse_loss(model(x.cuda()).squeeze(), y.cuda()).item()
    return loss/len(loader) 

# Get values from a normal distribution
def gauss_range(num_vals, bound = 3, sigma = 1, mu = 0, norm_factor=None):
    t_range = torch.arange(num_vals)
    norm_t_range = (bound*2)*((t_range-t_range.min())/(t_range.max()-t_range.min()))-bound
    gauss_values = 1/(sigma * torch.sqrt(torch.tensor(2 * np.pi))) * torch.exp( - (norm_t_range - mu)**2 / (2 * sigma**2) )
    gauss_values = gauss_values if norm_factor == None else gauss_values / (norm_factor*gauss_values.max())
    return gauss_values

# Normalization function
def norm(data, norm_factor=1, minmax_norm=True):
    if minmax_norm:
        return (data-data.min()[...,np.newaxis])/(norm_factor*data.max()[...,np.newaxis]-data.min()[...,np.newaxis])
    else:
        return (data)/(norm_factor*data.max()[...,np.newaxis])

def norm_np(data, norm_factor=1, axis=None):
    return (data-data.min(axis=axis)[...,np.newaxis])/(norm_factor*data.max(axis=axis)[...,np.newaxis]-data.min(axis=axis)[...,np.newaxis])

def norm_tensor(data, norm_factor=1, dim=None, minmax_norm=True):
    if minmax_norm:
        return (data- data.amin(dim=dim).unsqueeze(1))/(norm_factor*data.amax(dim=dim).unsqueeze(1)- data.amin(dim=dim).unsqueeze(1))
    else:
        return data/(norm_factor*data.amax(dim=dim).unsqueeze(1))

# Find the nearest element in a array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# Get a difference set from a loop set
def get_state_acion_from_loop_set(power_set, setting_set, spectrum_set, norm=900):
    samples = list(itertools.product(np.arange(len(power_set)), np.arange(len(power_set))))
    states = []
    actions = []
    for sample in samples:
        state = torch.cat([torch.tensor(spectrum_set[sample[0]]), torch.tensor(spectrum_set[sample[1]])])
        setting_difference = setting_set[sample[1]]-setting_set[sample[0]]
        power_difference = [power_set[sample[1]]-power_set[sample[0]]]
        action = torch.cat([torch.tensor(power_difference), torch.tensor(setting_difference)])
        actions.append(action)
        states.append(state)
    states = torch.stack(states)/900
    actions = torch.stack(actions)
    return states, actions

# Get a historic state action pairs from a set (i.e. loop)
def get_historic_states_actions(start_spectrum,spectrum_set, power_set, setting_set, agent=True):
    if agent:
        states = torch.tensor(np.concatenate([start_spectrum[:,1:].transpose().repeat(spectrum_set.shape[0],axis=0)/3500, spectrum_set/2000], axis=1)).float()
        actions = torch.tensor(np.array([np.concatenate([np.array([power_set[i]]),setting_set[i,:]]) for i in range(setting_set.shape[0])])).float()
        return states, actions
    else:
        state_action = torch.tensor(np.concatenate([start_spectrum[:,1:].transpose().repeat(spectrum_set.shape[0],axis=0)/3500, spectrum_set/2000], axis=1)).float()
        next_states = torch.tensor(np.array([np.concatenate([np.array([power_set[i]]),setting_set[i,:]]) for i in range(setting_set.shape[0])])).float()
        return state_action, next_states

# Get set from manual files:
def get_model_set(step_paths, base_path_settings, base_path_spectra, set_name,save=False):
    power_set = []
    setting_base_set = []
    spectrum_set = []
    for step in step_paths:
        spectrum = np.loadtxt(base_path_spectra+step[0])
        setting, power,norm = read_setting(base_path_settings+step[1])
        setting_base_set.append(setting[:,1])
        power_set.append(power)
        spectrum_set.append(spectrum[:,1]) 
        
    setting_base_set = np.array(setting_base_set).astype(float)
    power_set = np.array(power_set).astype(float)
    spectrum_set = np.array(spectrum_set).astype(float)
    if save:
        np.save('setting_base_{}'.format(set_name), setting_base_set)
        np.save('power_{}'.format(set_name), power_set)
        np.save('spectrum_{}'.format(set_name), spectrum_set)
    return setting_base_set, power_set, spectrum_set
        

# Get a loop set from data
def get_loop_set(setting_path, spectrum_path, start_time, set_name, save=False):
    setting_list = [file for file in listdir(setting_path) if "ipynb" not in file]
    setting_list.sort()
    spectrum_list = [file for file in listdir(spectrum_path) if file[-3:] == 'txt']
    spectrum_list_file_order = [int(file[11:19].replace('.','')) for file in spectrum_list if int(file[11:19].replace('.',''))>start_time]
    spectrum_list_file_order.sort()
    print(setting_list)
    setting_base_set = []
    setting_set = []
    power_set = []
    spectrum_set = []
    for i in range(len(setting_list)):
        setting_file = setting_list[i]
        spectrum_file = [spec for spec in spectrum_list if int(spec[11:19].replace('.','')) == spectrum_list_file_order[i]][0]
        spectrum= np.loadtxt(spectrum_path+'{}'.format(spectrum_file))
        x = spectrum[:,0]
        setting, power,norm = read_setting(setting_path+'{}'.format(setting_file))
        setting_interp = np.interp(x,setting[:,0],setting[:,1])
        
        setting_base_set.append(setting[:,1])
        setting_set.append(setting_interp)
        power_set.append(power)
        spectrum_set.append(spectrum[:,1]) 
        
    setting_set = np.array(setting_set).astype(float)
    power_set = np.array(power_set).astype(float)
    spectrum_set = np.array(spectrum_set).astype(float)
    settting_base_set = np.array(setting_base_set).astype(float)
    if save:
        np.save('setting_base_{}'.format(set_name), setting_base_set)
        np.save('power_{}'.format(set_name), power_set)
        np.save('setting_{}'.format(set_name), setting_set)
        np.save('spectrum_{}'.format(set_name), spectrum_set)
    return setting_set, power_set, spectrum_set, setting_base_set

# Get the norm value based on closest power value in a set
def get_norm_from_data(power, norm_path = '/home/bethke52/laser_data/historic_data/norm_power_historic.npy'):
    norm_power = np.load(norm_path)
    mean_norm_power = []
    for power in np.unique(norm_power[0,:]):
        mean_norm = np.mean(norm_power[1,np.where(power == norm_power[0,:])[0]])
        mean_norm_power.append([power, mean_norm])
    mean_norm_power = np.array(mean_norm_power)
    return mean_norm_power[np.where(find_nearest(mean_norm_power[:,0],power)==mean_norm_power[:,0])[0],1][0]
    
# Get the target from closest max intestity of a set
def get_target_from_max(start, spectrum_set_path='/home/bethke52/laser_data/historic_data/historic_spectra_set.npy', smooth=True):
    spectrum_set = np.load(spectrum_set_path)
    # Get step from max val target to max val start relation
    max_start = []
    for i in spectrum_set:
        max_start.append(gaussian_filter1d(i[0,:,1], sigma=6).max())

    max_start = np.array(max_start)
    
    max_val = gaussian_filter1d(start[:,1], sigma=6).max()
    step =  spectrum_set[np.where(find_nearest(max_start,max_val)==max_start)[0]][0,1]
    
    return step

# Get the target from closest max intestity of a set
def get_target_from_data(start, spectrum_set_path='/home/bethke52/laser_data/historic_data/historic_spectra_set.npy', smooth=True, return_step_start=False, normed=False):
    spectrum_set = np.load(spectrum_set_path)
    start = start[:,1]
    if smooth:
        start = gaussian_filter1d(start, sigma=6)
    if normed:
        start = norm(start)
        
    # Get step from max val target to max val start relation
    mean_abs_diff= []
    for i in spectrum_set:
        if normed: 
            mean_abs_diff.append(abs(norm(gaussian_filter1d(i[0,:,1], sigma=6))-start).mean())
        else:
            mean_abs_diff.append(abs(gaussian_filter1d(i[0,:,1], sigma=6)-start).mean())

    mean_abs_diff = np.array(mean_abs_diff)
    closest_idx = mean_abs_diff.argsort()
    step =  spectrum_set[closest_idx[0]][1]
    step_start =  spectrum_set[closest_idx[0]][0]
    
    if return_step_start:
        return step, step_start
    else:
        return step

# Get a small array of targets derived from a base loop target
def differ_targets(target, smooth=True, smooth_ramp=False, smooth_step=False, shift=50, dynamic_resampling=False):
    target = target.copy()
    target_smooth = gaussian_filter1d(target[:,1], sigma=6)
    if smooth:
        target[:,1] = target_smooth
    target_shift_left = np.concatenate([target[shift:,1], target[:shift,1]])
    target_shift_right = np.concatenate([ target[-shift:,1],target[:-shift,1],])
    sigma= 0.15
    gauss_1 = gauss_range(target[:,1].shape[0], sigma=sigma, mu=-0.2).detach().numpy()*50 + target[:,1] 
    gauss_2 = gauss_range(target[:,1].shape[0], sigma=sigma, mu=0).detach().numpy()*50 + target[:,1] 
    gauss_3 = gauss_range(target[:,1].shape[0], sigma=sigma, mu=-0.45).detach().numpy()*50 + target[:,1] 

    gauss_1 = (gauss_1/gauss_1.max()) * target[:,1].max()
    gauss_2 = (gauss_2/gauss_2.max()) * target[:,1].max()
    gauss_3 = (gauss_3/gauss_3.max()) * target[:,1].max()
    
    left_arrange =260-50
    right_arrange =440-50
    step_target = np.concatenate([np.zeros(512+left_arrange),np.ones(1024-left_arrange-right_arrange),np.zeros(512+right_arrange)])*350
    pointy_target = np.concatenate([np.zeros(193*3),np.arange(378),np.arange(378)[::-1],np.zeros(238*3-1)])*0.9
    ramp_target_lr =  np.concatenate([np.zeros(193*3+160),np.arange(2*378-160)[::-1],np.zeros(238*3-1)])*0.9
    ramp_target_lr_small =  np.concatenate([np.zeros(193*3+160),np.arange(2*378-160-120)[::-1],np.zeros(238*3-1+120)])*0.9
    ramp_target_rl = np.concatenate([np.zeros(193*3+160),np.arange(2*378-160),np.zeros(238*3-1)])*0.9
    
    if smooth_ramp:
        ramp_target_lr = gaussian_filter1d(ramp_target_lr, sigma=45)
        ramp_target_rl = gaussian_filter1d(ramp_target_rl, sigma=45)
        ramp_target_lr_small = gaussian_filter1d(ramp_target_lr_small, sigma=45)
    if smooth_step:
        step_target = gaussian_filter1d(step_target, sigma=45)
        

    target_labels = ["gauss_1","gauss_2","gauss_3", "target_shift_left", "target_shift_right", "target_smooth", "step_target", "pointy_target", "ramp_target_lr", "ramp_target_rl", "ramp_traget_lr_small"]
    targets =  [gauss_1,gauss_2,gauss_3, target_shift_left, target_shift_right, target_smooth, step_target, pointy_target, ramp_target_lr, ramp_target_rl, ramp_target_lr_small] 
    
    # Dynamic resampling
    if dynamic_resampling:
        non_zero_idx_step = np.where(norm(target[:,1]) > 0.002)[0]
        for i in range(6,len(targets)):
            non_zero_idx_target = np.where(norm(targets[i]) > 0.002)[0]
            target_non_zero_resample = resample(targets[i][non_zero_idx_target.min():non_zero_idx_target.max()], non_zero_idx_step.max()-non_zero_idx_step.min())
            target_resample = norm(target[:,1].copy())
            target_resample[non_zero_idx_step.min():non_zero_idx_step.max()] = norm(target_non_zero_resample)
            targets[i] = target_resample
            
    return np.array(targets), np.array(target_labels)

# Perform live-inference with multiple targets on a history-based agent
def multi_target_inference(start, agent, model_name = '', fixed=None, save_file=False, base_name="Friedrich_Bethke_test_setting",smooth_targets= True, smooth_settings=False, downsampling_rate=None, smooth_move_to_zero=False, smooth_states = False, norm_states = False, kernel_size=25, smooth_ramp=False, normed=False, dynamic_resampling=False, smooth_step=False, power_norm=False):
    # Get loop step
    if fixed == None:
        step = get_target_from_data(start, normed=normed)
    else: 
        step = np.loadtxt(fixed)
    
    # Get differed targets
    targets, target_labels = differ_targets(step, smooth_targets, smooth_ramp, dynamic_resampling=dynamic_resampling, smooth_step=smooth_step)

    # Creates states
    states = torch.tensor(np.concatenate([np.repeat(start[:,1][np.newaxis,...]/3500,targets.shape[0],axis=0),targets/2000], axis=1)).float().to(device)
    if smooth_states:
        states = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)(states)
    if norm_states:
        states = torch.cat([norm_tensor(states[:,:2048], dim=1),norm_tensor(states[:,2048:], dim=1)], dim=1)
    if downsampling_rate != None:
        states = F.interpolate(states.unsqueeze(0), (downsampling_rate*2)).squeeze(0)
    
    predicted_action = agent(states)
    if downsampling_rate != None:
        predicted_action = torch.cat([predicted_action[:,0].unsqueeze(1),F.interpolate(predicted_action[:,1:].unsqueeze(0), (2047)).squeeze(0)], dim=1)
    power = np.clip(predicted_action[:,0].cpu().detach().numpy(),0,1)
    if power_norm:
        power = power * 0.05
    wave_values = torch.clip(predicted_action[:,1:],0,1).cpu().detach().numpy()
    if smooth_settings:
        wave_values = np.clip(np.array([gaussian_filter1d(i,sigma=6) for i in wave_values]),0,1)
    if smooth_move_to_zero:
        wave_values = np.array([move_to_zero(i) for i in wave_values])
    norm = get_norm_from_data(power)
    if save_file:
        for i in range(wave_values.shape[0]):
            file_name = base_name+"_{}_{}_{}_{}_{}".format(target_labels[i],"fixed" if fixed != None else "notFixed", model_name, smooth_targets, dynamic_resampling)
            write_setting(power[i], wave_values[i], norm, file_name)
    return power, wave_values, norm, targets, target_labels

# Savety check based on smoothed setting: suggested thres range [0.05,0.07]
def eval_signal(signal, sigma=9, plot=False, relative=False, lower=-0.07, upper=0.07, figsize=(8,5), return_signal = False, title=None):
    smooth_sig = gaussian_filter1d(signal, sigma=sigma)
    if relative:
        smooth_sig_upper = smooth_sig*upper
        smooth_sig_lower = smooth_sig*lower
    else:
        smooth_sig_upper = smooth_sig+upper
        smooth_sig_lower = smooth_sig+lower
        
    usable = False if (signal > smooth_sig_upper).any() or (signal < smooth_sig_lower).any() else True
    
    if plot:
        x_vals_setting = np.load('setting_x_values.npy')
        plt.figure(figsize=figsize)
        plt.title("Classified as: {}".format("Save" if usable else "Dangerous"))
        plt.plot(x_vals_setting, smooth_sig, alpha=1., label="Smoothed wave")
        plt.plot(x_vals_setting, signal, alpha=1., label="Original wave")
        plt.plot(x_vals_setting, smooth_sig_upper, linestyle='--', alpha=0.5, label="Upper bound")
        plt.plot(x_vals_setting, smooth_sig_lower, linestyle='--', alpha=0.5, label="Lower bound")
        plt.grid(alpha=0.5)
        plt.legend()
        plt.ylabel("Normalized intensity intensity", size=12)
        plt.xlabel("Wavelength [nm]", size=12)
        plt.tight_layout()
        if title != None:
            plt.savefig(title, dpi=300)
        plt.show()
    if return_signal == False:
        return usable
    else:
        return smooth_sig

# Inference of difference based models
def difference_model_inference(agent, current_spectrum, target_spectrum, current_setting_wave, current_setting_power):
    x_spectrum = current_spectrum[:,0]
    state = torch.cat([torch.tensor(current_spectrum[:,1]), torch.tensor(target_spectrum)]).float() / 900
    predicted_action = agent(state.unsqueeze(0).to(device)).cpu().detach().numpy()[0]
    # Postprocessing: x vals to setting x vals
    setting_x_vals = np.load('setting_x_values.npy')
    predicted_power_setting = predicted_action[0]
    predicted_wave_setting = np.interp(setting_x_vals, x_spectrum, predicted_action[1:])
    next_setting_wave = np.clip(current_setting_wave[:,1] + predicted_wave_setting,0,1)
    next_setting_power = np.clip(current_setting_power + predicted_power_setting,0,1)
    return next_setting_wave,  next_setting_power

# Plot used for logging evaluation plots to W&B
def action_eval_plot(agent,num_plots, states, actions, log_writer = None, index = None, downsampling_rate = None, epoch=None):
    setting_x = np.load('/home/bethke52/laser_data/setting_x_values.npy')
    if downsampling_rate != None:    
        setting_x = signal.resample(setting_x,downsampling_rate)

    for i in range(num_plots):
        if index != None:    
            sample = index[i]
        else:
            sample = i
        state = states[sample].float()
        with torch.no_grad():
            predicted_action = agent(state.unsqueeze(0).to(device))
            
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(setting_x, actions[sample,1:].cpu().detach().numpy(),label='GT', alpha=0.5)
        ax.plot(setting_x, predicted_action[0,1:].cpu().detach().numpy(), label='Predicted', alpha=0.5)
        ax.set_ylim(-0.1,1.1)
        ax.legend()
        #ax[1].scatter(1,action[0].cpu().detach().numpy(), label='GT', alpha=0.5)
        #ax[1].scatter(1,predicted_action[0,0].cpu().detach().numpy(), label='Predicted', alpha=0.5)
        #x[1].set_ylim(0.01,0.04)
        #x[1].legend()
        #fig.tight_layout()
        if index != None:
            #plt.suptitle('Sample {}, Epoch {}'.format(index[i].item(), epoch))
            plt.title('Sample {}, Epoch {}'.format(index[i].item(), epoch))
        else:
            #plt.suptitle('Sample {}, Epoch {}'.format(i, epoch))
            plt.title('Sample {}, Epoch {}'.format(i, epoch))
        if log_writer == None:
            plt.show()
        else:        
            #log_writer[1].log({"Output on sample {}".format(sample): log_writer[1].Image(plt), "Epoch": epoch})
            log_writer[1].log({"Output on sample {}".format(sample): log_writer[1].plot.line_series(
                       xs=setting_x, 
                       ys=[actions[sample,1:].cpu().detach().numpy(), predicted_action[0,1:].cpu().detach().numpy()],
                       keys=["GT", "Predicted"],
                       title='Sample {}, Epoch {}'.format(index[i].item(), epoch),
                       xname="wavelength"), "Epoch": epoch})


            plt.close(fig)
        plt.close(fig)

# Mazzler software hole functionality
def hole(wave, hole_depth, hole_width, hole_position, mazzler_fix=True):
    omega = wave
    #C = 100592.4376945696
    k = hole_depth
    delta_lambda_1 = hole_width
    lambda_1 = hole_position
    if mazzler_fix and lambda_1 < 800:
        lambda_1 += 10
    #omega_1 = (2* np.pi * C) / lambda_1
    omega_1 = lambda_1
    xi_1 = delta_lambda_1/(2*lambda_1)
    delta_omega_1 = omega_1 * (xi_1 - xi_1**3)/2
    return 1- k * np.exp(-((omega - omega_1)/delta_omega_1)**2)

# A function to smooth outliers and remove downward spikes
def move_to_zero(wave):
    return norm(np.clip(wave-np.median(wave),0,1))

# Reverse engineered feedback loop 
def re_loop(start, loop_set_name):
    setting_loop= np.load('/home/bethke52/laser_data/live_data/setting_base_{}.npy'.format(loop_set_name))
    spectrum_loop = np.load('/home/bethke52/laser_data/live_data/spectrum_{}.npy'.format(loop_set_name))
    power_loop = np.load('/home/bethke52/laser_data/live_data/power_{}.npy'.format(loop_set_name))
    spectrum_start = np.loadtxt('/home/bethke52/laser_data/{}'.format(start))
    x = spectrum_start[:,0]
    filter_= [+630.000000E+0,+7.329000E+0,+680.000000E+0,+7.329000E+0,+690.000000E+0,+7.353000E+0,+700.000000E+0,+7.377000E+0,+710.000000E+0,+7.399000E+0,+720.000000E+0,+7.418000E+0,+730.000000E+0,+7.436000E+0,+740.000000E+0,+7.460000E+0,+750.000000E+0,+7.486000E+0,+760.000000E+0,+7.510000E+0,+770.000000E+0,+7.541000E+0,+780.000000E+0,+7.572000E+0,+790.000000E+0,+7.603000E+0,+800.000000E+0,+7.639000E+0,+810.000000E+0,+7.676000E+0,+820.000000E+0,+7.749000E+0,+830.000000E+0,+7.790000E+0,+840.000000E+0,+7.833000E+0,+850.000000E+0,+7.877000E+0,+860.000000E+0,+7.922000E+0,+870.000000E+0,+7.968000E+0,+880.000000E+0,+8.015000E+0,+890.000000E+0,+8.060000E+0,+900.000000E+0,+8.109000E+0,+910.000000E+0,+8.158000E+0,+920.000000E+0,+8.201000E+0,+970.000000E+0,+8.201000E+0]
    filter_interp = np.interp(x,filter_[::2],filter_[::-2])

    re_init_setting = norm(uniform_filter1d((filter_interp*spectrum_start[:,1]**0.55),size=25), minmax_norm=False)
    re_settings = []
    re_settings.append(re_init_setting)
    for i in range(spectrum_loop.shape[0]):
        next_setting = norm((re_settings[i]+0.05*(2*norm(uniform_filter1d(spectrum_loop[i]-spectrum_start[:,1], size=15))-1)))
        re_settings.append(next_setting)
    for i in range(len(re_settings)-1):
        plt.title("Step {} Power {}".format(i, power_loop[i]))
        plt.plot(re_settings[i], label="Reverse")
        plt.plot(setting_loop[i], label="GT")
        plt.legend()
        #plt.xlim(800,1080)
        #plt.ylim(0.9,1.1)
        plt.show()
    print("#####################################################################")

# Loss that focusses on the power parameter in the action space
def power_loss(predicted_action, action):
    return torch.nn.MSELoss()(predicted_action[:,0], action[:,0]) + torch.nn.MSELoss()(predicted_action[:,1:], action[:,1:])

# Modify signals with multiple dangerous holes
def multi_hole(signal, num_holes = 2):
    x_vals = np.load('/home/bethke52/laser_data/setting_x_values.npy')
    hole_wave = np.zeros(x_vals.shape[0])
    signal_mod = signal.copy()
    for i in range(num_holes):
        range_idx = np.where(signal > 0.4)[0]
        range_lower = x_vals[range_idx.min()]
        range_upper = x_vals[range_idx.max()]
        pos = np.random.uniform(range_lower, range_upper)
        width = np.random.uniform(10,15)
        depth = np.random.uniform(0.8,0.9)
        
        depth_scale = 1.4 - signal[np.where(find_nearest(x_vals, pos) == x_vals)[0][0]].item()
        depth = depth*depth_scale
        
        signal_mod = norm(signal_mod * hole(x_vals, depth, width, pos,mazzler_fix=False))
    return signal_mod

# Modify signals with sinus noise + minor gaussian noise
def high_freq_signals(sigs, fac = 0.05,freq_lim = 25, plot_noise = False, num_sin = 3):
    num_sig = sigs.shape[0]
    size = sigs.shape[1]
    x = np.arange(size)[np.newaxis,...].repeat(num_sig,axis=0) / size
    
    
    rand_sin = np.array([np.sin(2*np.pi*x*np.random.uniform(7,freq_lim,size=(num_sig,1))+np.random.uniform(0,2*np.pi)) for i in range(num_sin)])
    noise = rand_sin.prod(axis=0) + np.random.normal(size=(num_sig,size))*np.random.uniform(0,0.05)

    if plot_noise:
        [plt.plot(i) for i in noise]
        plt.show()
    
    mod_sigs = sigs + noise * fac
    return mod_sigs

# Add holes and sinus+gauss noise to signals
def add_danger_noise(actions):
    hole_actions = []
    for i in range(actions.shape[0]):
        hole_action = multi_hole(actions[i], num_holes=np.random.randint(1,5))
        hole_action = high_freq_signals(hole_action[np.newaxis,...],num_sin=np.random.randint(0,3))[0]
        hole_actions.append(hole_action)
    hole_actions = np.array(hole_actions)
    return hole_actions

# Create bad examples for mazzler actions based on previous ones and random noise
def create_bad_examples(actions, size=2047, num_mod=1000, num_noise=1000):
    actions_idx = np.random.randint(0,actions.shape[0],num_mod)
    danger_actions = add_danger_noise(actions[actions_idx])
    uniform_noise = np.array([np.random.uniform(np.random.uniform(-1,0),np.random.uniform(0,1),size=size) for i in range(num_noise//2)])
    normal_noise = np.array([ np.random.normal(np.random.uniform(-0.5,0.5),np.random.uniform(0,1),size=size) for i in range(num_noise//2)])
    return np.concatenate((danger_actions, uniform_noise, normal_noise))

# Get actions for piecewise linear spline
def get_pwl_actions(actions, base_waves=None):
    x_vals_setting = np.load('/home/bethke52/laser_data/setting_x_values.npy')
    x_vals_spectrum = np.load('/home/bethke52/laser_data/spectrum_x_values.npy')
    if base_waves==None:
        waves = actions[:,1:].cpu().detach().numpy()
    else:
        waves= base_waves.cpu().detach().numpy()
    pos_thres = 0.01
    waves_pos = np.array([(x_vals_setting[np.where(waves[i] > pos_thres)[0].min()], x_vals_setting[np.where(waves[i] > pos_thres)[0].max()]) for i in range(waves.shape[0])])

    thres = 10
    num_knots = 40
    knots = np.linspace(waves_pos[:,0].min()-thres, waves_pos[:,1].max()+thres,num_knots)

    mean_pos = np.mean((waves_pos[:,1].max(),waves_pos[:,0].min()))
    edge = (mean_pos-waves_pos[:,0].min())
    focus_knots = np.concatenate((
        np.linspace(waves_pos[:,0].min()-thres, mean_pos-(edge/2),num_knots//4),
        np.linspace(mean_pos-(edge/2),mean_pos+(edge/2),num_knots//2),
        np.linspace(mean_pos+(edge/2), waves_pos[:,1].max()+thres,num_knots//4)))
    knots = focus_knots

    x_knots = np.array([find_nearest(x_vals_setting, i)for i in knots])
    x_knots = np.concatenate((np.array([x_vals_setting[0]]), x_knots, np.array([x_vals_setting[-1]])))
    xn = torch.tensor(x_vals_setting).float().to(device)
    xp = torch.tensor(x_knots).float().to(device)

    knots_idx = [np.where(i == x_vals_setting)[0][0] for i in x_knots]

    pwl_actions = torch.cat((actions[:,:1], actions[:,1:][:,knots_idx]), dim=1)   
    return pwl_actions, xn, xp

# Get surrogate states for piecewise linear spline
def get_pwl_surrogate_states(surrogate_states, base_waves=None):
    actions = surrogate_states[:,int(surrogate_states.shape[1]/2):]
    pwl_actions, xn, xp = get_pwl_actions(actions, base_waves)
    pwl_surrogate_states = torch.cat((surrogate_states[:,:int(surrogate_states.shape[1]/2)], pwl_actions), dim=1)
    return pwl_surrogate_states, xn, xp


# A function to load live experience from source files, matching on integer in name extracted by splitting on '_' and [0]
def get_live_exp_data(base_path_settings, base_path_spectra, start_spectrum, smooth_states=False, kernel_size = 25, norm_states=False, norm_power=False, norm_power_max=0.05):
    
    setting_list = np.array([file for file in listdir(base_path_settings) if'ipynb' not in file])
    spectra_list = np.array([file for file in listdir(base_path_spectra) if 'start' not in file and 'ipynb' not in file] )

    spectra_list = spectra_list[np.argsort([int(file.split('_')[0]) for file in spectra_list])][np.newaxis,...]
    setting_list = setting_list[np.argsort([int(file.split('_')[0]) for file in setting_list])][np.newaxis,...]
    
    spectrum_setting = np.concatenate((spectra_list.T, setting_list.T), axis=1)
    
    
    setting_loop, power_loop, spectrum_loop = get_model_set(spectrum_setting, base_path_settings, base_path_spectra, '', save=False)
    spectrum_start = np.loadtxt(base_path_spectra+start_spectrum)

    # norm step
    setting_loop= norm_np(setting_loop, axis=1)

    states, actions = get_historic_states_actions(spectrum_start,spectrum_loop,power_loop,setting_loop)

    if smooth_states:
        states = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)(states)
    if norm_states:
        states = torch.cat([norm_tensor(states[:,:2048], dim=1),norm_tensor(states[:,2048:], dim=1)], dim=1)
    if norm_power:
        actions[:,0] = actions[:,0]/norm_power_max
    return states, actions