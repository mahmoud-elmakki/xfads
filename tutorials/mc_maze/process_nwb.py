import yaml
import torch

from itertools import product
from nlb_tools.nwb_interface import NWBDataset



def main():
    
    datapath = 'data/000128/sub-Jenkins/'
    dataset = NWBDataset(datapath)
    save_root_path = 'data/'

    # Extract neural data and lagged hand velocity
    binsize = 5
    n_neurons = 182
    dataset.resample(binsize)

    # We want total trial length of 900ms ...
    start = -450
    end = 450
    #... which is 90 time bins
    trial_length = (end - start) // binsize

    # Extract neural data
    trial_info = dataset.trial_info  # .dropna()
    
    # Trials aligned around the movement_onset time bin
    trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range=(start, end))
    n_trials = trial_data.shape[0] // trial_length
    
    y = []
    target = []

    for trial_id, trial in trial_data.groupby('trial_id'):
        trial_id_trial_info = trial_info[trial_info['trial_id'] == trial_id]

        y_heldin_t = torch.tensor(trial.spikes.values)
        y_heldout_t = torch.tensor(trial.heldout_spikes.values)
        y_t = torch.concat([y_heldin_t, y_heldout_t], dim=-1)
        y.append(y_t.reshape(1, trial_length, n_neurons))

        target.append(torch.tensor(trial.hand_vel.values).reshape(1, trial_length, 2))

    y = torch.concat(y, dim=0)
    target = torch.concat(target, dim=0)

    train_data, valid_data, test_data = {}, {}, {}
    seq_len = y.shape[1]
    n_neurons = y.shape[-1]
    n_valid_trials = 574

    train_data['y_obs'] = y[:-n_valid_trials]
    train_data['velocity'] = target[:-n_valid_trials]
    train_data['n_neurons_enc'] = y.shape[-1]
    train_data['n_neurons_obs'] = y.shape[-1]
    train_data['n_time_bins_enc'] = seq_len

    valid_data['y_obs'] = y[-n_valid_trials:-n_valid_trials // 2]

    valid_data['velocity'] = target[-n_valid_trials:-n_valid_trials // 2]
    valid_data['n_neurons_enc'] = n_neurons
    train_data['n_neurons_obs'] = n_neurons
    valid_data['n_time_bins_enc'] = seq_len

    test_data['y_obs'] = y[-n_valid_trials // 2:]
    test_data['velocity'] = target[-n_valid_trials // 2:]
    test_data['n_neurons_enc'] = n_neurons
    test_data['n_neurons_obs'] = n_neurons
    test_data['n_time_bins_enc'] = seq_len
    
    torch.save(train_data, save_root_path + f'data_train_{binsize}ms.pt')
    torch.save(valid_data, save_root_path + f'data_valid_{binsize}ms.pt')
    torch.save(test_data, save_root_path + f'data_test_{binsize}ms.pt')
    
    print('Data splits (train/valid/test) saved into the "data" folder.')


if __name__ == '__main__':
    main()