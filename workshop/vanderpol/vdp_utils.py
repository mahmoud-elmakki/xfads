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