import numpy as np
import matplotlib.pyplot as plt
import random

import torch


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
        [axs[i].axvline(55, linestyle='--', color='red', alpha=0.5) for i in range(len(ex_trials))]
    
    [axs[i].set_title(f'trial {ex_trials[i]}', fontsize=8) for i in range(len(ex_trials))]
    
    [axs[i].plot(latents[i, :, n], color=color, linewidth=0.8, alpha=0.8, label=regime if i == 0 and n == 0 else '')
     for i in range(n_trials) for n in range(n_latents)]

    [axs[i].set_xlim(0, n_bins) for i in range(n_trials)]
    
    ymin, _ = axs[len(ex_trials)-1].get_ylim()
    
    axs[len(ex_trials)-1].annotate('prediction\nstarts', xy=(55, ymin), xytext=(5, ymin+ymin),
             arrowprops=dict(facecolor='gray', arrowstyle='->', alpha=0.5),
             fontsize=8, ha='center', alpha=0.5)
    
    fig.tight_layout()