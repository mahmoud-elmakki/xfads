import numpy as np
import matplotlib.pyplot as plt
import random

import torch



def gen_unique_rand_ints(mn, mx, n=4):
    
    unique_numbers = []
    
    while len(unique_numbers) < n:
        number = random.randint(mn, mx)
        
        if number not in unique_numbers:
            unique_numbers.append(number)
            
    return unique_numbers


def plot_rastor(data, ex_trials, top_n_neurons, cfg):
    
    data = data.clone()
        
    with torch.no_grad():

        plt.figure(figsize=(16, 6))
        
        n_trials_to_plot = 4
        n_time_bins = data.shape[1]

        fig, axes = plt.subplots(ncols=n_trials_to_plot, figsize=(14, 6))
        fig.suptitle('data trials')
        
        vmin, _ = torch.min(data[ex_trials].flatten(), dim=0)
        vmax, _ = torch.max(data[ex_trials].flatten(), dim=0)

        for ax, trial in zip(axes, ex_trials):

            cax = ax.imshow(data[trial].T[:top_n_neurons], cmap='viridis', interpolation='none', aspect='auto')
            
            cbar = fig.colorbar(cax, ax=ax, shrink=0.2)
            cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
            if fig.get_axes().index(ax) == 0:
                cbar.set_label('spike counts')
                
            ax.set_title(f'trial {trial+1}', fontsize=10)
            
            x_ticks = ax.get_xticks()
            x_labels = x_ticks * cfg.bin_sz_ms
            ax.set_xticks(x_ticks, x_labels)
            
            ax.set_xlim(0, n_time_bins)
            
            ax.spines['right'].set_visible(False)

        #  make the y-axis unvisible exept for the first plot.
        [axes[i].yaxis.set_visible('false') for i in range(1, n_trials_to_plot)]
        [axes[i].set_yticks([]) for i in range(1, n_trials_to_plot)]
        
        axes[0].set_xlabel('time (ms)')
        axes[0].set_ylabel('neurons')
        
        fig.tight_layout()

        plt.show()


def calculate_data_psth(trials, neurons):
    
    n_bins = n_time_bins
    bin_size = cfg.bin_sz  # Assuming bin size of 10 ms
    
    psths = []

    # trials x time bins
    for neuron in neurons:
    
        single_neuron = trials[:, :, neuron]

        # Calculate the PSTH by averaging across trials
        psth = torch.mean(single_neuron, axis=0) / bin_size
        psths.append(psth)
        
    return np.array(psths)


def calculate_model_psth(trials, sample, neurons):
    
    n_bins = n_time_bins
    bin_size = cfg.bin_sz
    
    trials = torch.mean(trials, dim=0)
    psths = []
    
    with torch.no_grad():

        # trials x time bins
        for neuron in neurons:
            
            single_neuron = trials[sample, :, :, neuron]

            # Calculate the PSTH by averaging across trials
            psth = torch.mean(single_neuron, axis=0)
            psths.append(psth)
        
    return np.array(psths)


def order_neurons(trials, latents, trial, latent):

    with torch.no_grad():

        trial_to_reorder = trials[trial]
        latent_to_compare = torch.mean(latents[:, 0, :, latent], dim=0)

        # Check for NaN or infinite values and clean the data if necessary
        trial_to_reorder = np.nan_to_num(trial_to_reorder, nan=0.0, posinf=0.0, neginf=0.0)
        latent_to_compare = np.nan_to_num(latent_to_compare, nan=0.0, posinf=0.0, neginf=0.0)

        def calc_corrcoef(x, y):
            with np.errstate(divide='ignore', invalid='ignore'):
                corr = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr):
                return 0.0
            return corr

        correlations = np.array([calc_corrcoef(trial_to_reorder[:, i], latent_to_compare) for i in range(trial_to_reorder.shape[1])])

        correlation_df = pd.DataFrame({'neuron': range(1, trial_to_reorder.shape[1] + 1), 'correlation': correlations})

        # Order the DataFrame by the absolute value of correlations
        ordered_correlation_df = correlation_df.reindex(correlation_df.correlation.abs().sort_values(ascending=False).index)

        return ordered_correlation_df.to_numpy(), ordered_correlation_df.to_numpy().T[0]
    
    
def plot_latent_trajectory(data, latents, ex_trials, latent_idx=0):

    with torch.no_grad():

        fig, axes = plt.subplots(len(ex_trials), 1, figsize=(6, 8))

        fig.suptitle(f'latent {latent_idx+1}\n\n\n\n')

        for i, ax in enumerate(axes.flat):    

            ax.plot(latents[i, :, latent_idx], color='red', alpha=0.4, label='inferred' if i == 0 else '')
            ax.plot(data[i, :, latent_idx], color='black', alpha=0.8, label='real' if i == 0 else '')

            ax.set_title(f'trial {ex_trials[i]}', fontsize=8)
            ax.set_xlabel('time bins' if i == len(ex_trials)-1 else '')
            #ax.set_ylabel('1st dim' if i == 0 else '')
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

        plt.tight_layout()

        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=1, fontsize=8)

        plt.show()
        
        
def plot_z_2d(fig, axs, ex_trials, latents, color, regime):
    
    n_trials, n_bins, n_latents = latents.shape
    
    fig.subplots_adjust(hspace=0)
    
    if regime == 'prediction':
        [axs[i].axvline(22, linestyle='--', color='red', alpha=0.5) for i in range(len(ex_trials))]
    
    [axs[i].set_title(f'trial {ex_trials[i]}', fontsize=8) for i in range(len(ex_trials))]
    
    [axs[i].plot(latents[i, :, n], color=color, linewidth=0.8, alpha=0.8, label=regime if i == 0 and n == 0 else '')
     for i in range(n_trials) for n in range(n_latents)]

    [axs[i].set_xlim(0, n_bins) for i in range(n_trials)]
    
    ymin, _ = axs[len(ex_trials)-1].get_ylim()
    
    axs[len(ex_trials)-1].annotate('prediction\nstarts', xy=(22, ymin), xytext=(5, ymin+ymin),
             arrowprops=dict(facecolor='gray', arrowstyle='->', alpha=0.5),
             fontsize=8, ha='center', alpha=0.5)
    
    fig.tight_layout()