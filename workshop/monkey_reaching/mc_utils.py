import numpy as np
import pandas as pd
import seaborn as sns
import math
import random

from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import torch



def plot_z_samples(fig, axs, data, trial_indcs, move_onset, n_bins_bhv, color_map_list):
    
    samples = data[:, trial_indcs, ..., -3:]
    n_samples, n_trials, n_bins, n_neurons = samples.shape
    
    fig.subplots_adjust(hspace=0)
    
    [axs[i].axvline(move_onset, linestyle='--', color='gray') for i in range(n_trials)]
    
    [axs[i].axis('off') for i in range(n_trials-1)]
    axs[-1].yaxis.set_visible(False)
    axs[-1].spines['left'].set_visible(False)
    axs[-1].spines['right'].set_visible(False)
    axs[-1].spines['top'].set_visible(False)
    
    [axs[i].plot(samples[j, i, :, n], color=color_map_list[n](j), linewidth=0.5, alpha=0.4)
     for i in range(n_trials) for j in range(samples.shape[0]) for n in range(n_neurons)]
    
    [axs[i].set_title(f'trial {trial_indcs[i]+1}', fontsize=7) for i in range(n_trials)]
    
    [axs[i].set_xlim(0, n_bins) for i in range(n_trials)]
    
    fig.tight_layout()
    
    
def plot_spikes(spikes, axs):
    
    n_bins = spikes.shape[0]
    n_neurons = spikes.shape[1]

    # fig, axs = plt.subplots(figsize=(6, 3))
    _, indices = torch.sort(spikes.mean(dim=0))
    spikes = spikes[:, indices][..., :]

    for n in range(n_neurons):
        time_ax = np.arange(n_bins)
        neuron_spikes = spikes[:, n]
        neuron_spikes[neuron_spikes > 0] = 1
        neuron_spikes = neuron_spikes * time_ax
        neuron_spikes = neuron_spikes[neuron_spikes > 0]

        axs.scatter(neuron_spikes, 0.5 * n * np.ones_like(neuron_spikes), marker='o', color='black', s=4,
                    edgecolors='none')
        
        
def plot_z_2d(fig, axs, data, trial_indcs, cfg, color, regime):
    
    samples = data[:, trial_indcs, ..., -3:]
    n_samples, n_trials, n_bins, n_neurons = samples.shape
    
    fig.subplots_adjust(hspace=0)
    
    [axs[i].axvline(cfg.move_onset, linestyle='--', color='gray') for i in range(n_trials)]
    
    if regime == 'prediction':
        [axs[i].axvline(cfg.n_bins_bhv, linestyle='--', color='red') for i in range(len(trial_indcs))]
    
    [axs[i].axis('off') for i in range(n_trials-1)]
    
    axs[-1].yaxis.set_visible(False)
    axs[-1].spines['left'].set_visible(False)
    axs[-1].spines['right'].set_visible(False)
    axs[-1].spines['top'].set_visible(False)
    
    axs[-1].set_xlabel('time bins')
    
    [axs[i].set_title(f'trial {trial_indcs[i]}', fontsize=8) for i in range(n_trials)]
    
    [axs[i].plot(torch.tensor(gaussian_filter1d(torch.mean(samples[:, i, :, n], dim=0), sigma=2, axis=0)), color=color, linewidth=0.8, alpha=0.8, label=regime if i == 0 and n == 0 else '')
     for i in range(n_trials) for n in range(n_neurons)]

    [axs[i].set_xlim(0, n_bins) for i in range(n_trials)]
    #[axs[i].set_ylim(-12, 12) for i in range(n_trials)]
    
    fig.tight_layout()
    
    
def plot_rastor(data, latents, trial_list, top_n_neurons, cfg, regime='real', order=False):
    
    data = data.clone()
    latents = latents.clone()
    
    if order == True:
        
        for trial in trial_list:
            # Get the indices of the ordered neurons based on their contribution to the first principle latent dimension
            ordered_correlations, ordered_neurons = order_neurons(data, latents, trial=trial, latent=0)
            # just reorder the neurons in the trials we want to plot.
            data[trial] = data[trial, :, ordered_neurons-1]
        
    with torch.no_grad():

        plt.figure(figsize=(16, 6))
        
        n_trials_to_plot = 4

        fig, axes = plt.subplots(ncols=n_trials_to_plot, figsize=(9, 4))
        fig.suptitle('generated trials' if regime in ['filtering', 'smoothing', 'prediction'] else 'real trials', fontsize=10)
        
        vmin, _ = torch.min(data[trial_list].flatten(), dim=0)
        vmax, _ = torch.max(data[trial_list].flatten(), dim=0)

        for ax, trial in zip(axes, trial_list):

            cax = ax.imshow(data[trial].T[:top_n_neurons], cmap='viridis', interpolation='none', aspect='auto')
            
            cbar = fig.colorbar(cax, ax=ax, shrink=0.2)
            cbar.mappable.set_clim(vmin=vmin, vmax=vmax)
            if fig.get_axes().index(ax) == 0:
                cbar.set_label('spike counts')
                
            ax.set_title(f'trial {trial+1}', fontsize=10)
            
            x_ticks = ax.get_xticks()
            x_labels = x_ticks * cfg.bin_sz_ms
            ax.set_xticks(x_ticks, x_labels)
            
            ax.set_xlim(0, data.shape[1])
            
            ax.spines['right'].set_visible(False)
            
            if regime == 'prediction':
                ax.axvline(x=cfg.n_bins_bhv, color='y', linestyle='--')  # mark prediction start
            ax.axvline(x=cfg.move_onset, color='r', linestyle='--')  # mark movement onset
            
            if trial==trial_list[0]:
                
                if regime == 'prediction':
                    ax.annotate('pred\nstarts', xy=(cfg.n_bins_bhv, 0), xytext=(11-5-3, -20),
                         arrowprops=dict(facecolor='black', arrowstyle='->'),
                         fontsize=8, ha='center')
            
                ax.annotate('move\nonset', xy=(cfg.move_onset, 0), xytext=(11+5, -20),
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     fontsize=8, ha='center')
            
        #  make the y-axis unvisible exept for the first plot.
        [axes[i].yaxis.set_visible('false') for i in range(1, n_trials_to_plot)]
        [axes[i].set_yticks([]) for i in range(1, n_trials_to_plot)]
        
        axes[0].set_xlabel('time (ms)')
        axes[0].set_ylabel('neurons' if order == False else 'ordered neurons')
        
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


def calculate_model_psth(trials, neurons, sample=0):
    
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