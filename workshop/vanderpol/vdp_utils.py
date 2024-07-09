import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import torch


def order_neurons(trials, latents, trial, latent):

    with torch.no_grad():

        trial_to_reorder = trials[trial].cpu()
        latent_to_compare = latents[trial, :, latent].cpu()

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
    

def plot_rastor(data, cfg, trial_list=[0, 1, 2, 3], top_n_neurons=10, regime='real', order=False, latents=None):
    
    data = data.clone()
    
    if order == True:
        latents = latents.clone()
        
        for trial in trial_list:
            # Get the indices of the ordered neurons based on their contribution to the first principle latent dimension
            ordered_correlations, ordered_neurons = order_neurons(data, latents, trial=trial, latent=0)
            # just reorder the neurons in the trials we want to plot.
            data[trial] = data[trial, :, ordered_neurons-1]
        
    with torch.no_grad():

        plt.figure(figsize=(16, 6))
        
        n_trials_to_plot = len(trial_list)

        fig, axes = plt.subplots(ncols=n_trials_to_plot, figsize=(9, 4))
        fig.suptitle('generated trials' if regime in ['filtering', 'smoothing', 'prediction'] else 'real trials', fontsize=12)
        
        vmin, _ = torch.min(data[trial_list].flatten(), dim=0)
        vmax, _ = torch.max(data[trial_list].flatten(), dim=0)

        for ax, trial in zip(axes, trial_list):

            cax = ax.imshow(data[trial].T[:top_n_neurons].cpu(), cmap='viridis', interpolation='none', aspect='auto')
            
            cbar = fig.colorbar(cax, ax=ax, shrink=0.2)
            cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
            if fig.get_axes().index(ax) == 0:
                cbar.set_label('spike counts')
                
            ax.set_title(f'trial {trial+1}', fontsize=8)
            
            x_ticks = ax.get_xticks()
            x_labels = x_ticks * cfg.bin_sz_ms
            ax.set_xticks(x_ticks, x_labels)
            
            ax.set_xlim(0, data.shape[1])
            
            ax.spines['right'].set_visible(False)
            
            if regime == 'prediction':
                ax.axvline(x=cfg.n_bins_bhv, color='r', linestyle='--')  # mark prediction start
            
            if trial==trial_list[0]:
                
                if regime == 'prediction':
                    
                    ymin, ymax = ax.get_ylim()
    
                    ax.annotate(
                        'prediction\nstarts',
                        xy=(cfg.n_bins_bhv, 0), xytext=(cfg.n_bins_bhv*0.4, 0-top_n_neurons*0.05),
                        arrowprops=dict(facecolor='gray', arrowstyle='->', alpha=0.7),
                        fontsize=8, ha='center', alpha=1)
            
        #  make the y-axis unvisible exept for the first plot.
        [axes[i].yaxis.set_visible('false') for i in range(1, n_trials_to_plot)]
        [axes[i].set_yticks([]) for i in range(1, n_trials_to_plot)]
        
        axes[0].set_xlabel('time (ms)')
        axes[0].set_ylabel('neurons' if order == False else 'ordered neurons')
        
        fig.tight_layout()

        plt.show()
        
        
def plot_latent_trajectory(data, latents, ex_trials, latent_idx=0):

    with torch.no_grad():

        fig, axes = plt.subplots(len(ex_trials), 1, figsize=(6, 8))

        #fig.suptitle(f'latent {latent_idx+1}\n\n\n\n')
        fig.suptitle(f'first principal latent\n\n\n\n' if latent_idx == 0 else f'second principal latent\n\n\n\n')

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
        
        
def plot_z_2d(fig, axs, ex_trials, latents, cfg, color, regime):
    
    n_trials, n_bins, n_latents = latents.shape
    
    fig.subplots_adjust(hspace=0)
    
    if regime == 'prediction':
        [axs[i].axvline(cfg.n_bins_bhv, linestyle='--', color='red', alpha=0.5) for i in range(len(ex_trials))]
    
    [axs[i].set_title(f'trial {ex_trials[i]}', fontsize=8) for i in range(len(ex_trials))]
    
    [axs[i].plot(latents[i, :, n], color=color, linewidth=0.8, alpha=0.8, label=regime if i == 0 and n == 0 else '')
     for i in range(n_trials) for n in range(n_latents)]

    [axs[i].set_xlim(0, n_bins) for i in range(n_trials)]
    
    ymin, ymax = axs[len(ex_trials)-1].get_ylim()
    
    axs[len(ex_trials)-1].annotate(
                'prediction\nstarts',
                xy=(cfg.n_bins_bhv, ymin), xytext=(cfg.n_bins_bhv*0.4, ymin-np.abs((ymax-ymin)*0.3)),
                arrowprops=dict(facecolor='gray', arrowstyle='->', alpha=0.5),
                fontsize=8, ha='center', alpha=0.5)
    
    fig.tight_layout()