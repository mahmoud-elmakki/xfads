import numpy as np
import math
import argparse

import torch

from itertools import product
from nlb_tools.nwb_interface import NWBDataset



def main():
    
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="")

    # Add arguments
    parser.add_argument("data_folder_path", type=str, help="The path to the NWB data folder")
    parser.add_argument("test_split", type=float, help="Precentage of validation + test trials")
    parser.add_argument("binsize", type=int, help="The bin size you  want to resample the data to")
    parser.add_argument("align_event", type=str, help="The event to align the trials around")
    parser.add_argument("bins_before", type=int, help="")
    parser.add_argument("bins_after", type=int, help="")
    parser.add_argument("obj", type=str, help="")
    parser.add_argument("mes", type=str, help="Behavior (vel, pos, etc)")
    
    # Parse arguments
    args = parser.parse_args()

    bins_before = args.bins_before
    bins_after = args.bins_after
    n_bins = bins_before + bins_after
    obj = args.obj
    mes = args.mes

    datapath = args.data_folder_path
    dataset = NWBDataset(datapath)
    
    # Extract neural data and lagged hand velocity.
    binsize = args.binsize #ms
    dataset.resample(binsize)

    trial_info = dataset.trial_info

    # Combining the number of columns in the 'spikes' field with those in the 'heldout_spikes' field gives the total number of neurons.
    n_neurons = dataset.data.spikes.values.shape[1] + dataset.data.heldout_spikes.values.shape[1]
    n_null_trials = trial_info.isnull().sum()['success']
    print(f'number of neurons: {n_neurons}')
    print(f'total number of trials: {len(trial_info)}')
    print(f"number of null trials: {n_null_trials}")
    
    trial_data = dataset.make_trial_data()
    trials = [trial[1] for trial in trial_data.groupby('trial_id')]
    label_cols = [col for col in trial_data.columns if any(_ in col for _ in ['x', 'y'])]
    
    # Find unique conditions
    maze_conds = [cond for cond in trial_info.set_index(['trial_type', 'trial_version']).index.unique().tolist() if not any(math.isnan(x) for x in cond)]

    orig_conds = {}
    #simp_conds = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}

    # Loop over conditions and compute average trajectory
    for cond_idx, cond in enumerate(maze_conds):
        # Find trials in condition
        mask = np.all(dataset.trial_info[['trial_type', 'trial_version']] == cond, axis=1)
        trial_d = dataset.make_trial_data(ignored_trials=(~mask))
        orig_conds[cond_idx] = trial_d.trial_id.drop_duplicates().values
        #simp_conds[get_simple_cond(math.degrees(reach_angle) + 360 / 2)].append(trial_d.trial_id.drop_duplicates().values)

    maze_conds = torch.tensor(maze_conds)
    #simp_conds = {key: np.concatenate(value) for key, value in simp_conds.items()}
    maze_conds = torch.tensor(maze_conds)

    print(maze_conds.shape)
    print(orig_conds.keys())
    
    '''
    # Align the trials arount the move_onset bin with offsets before and after that bin.
    y = []
    labels = []

    bins_before_move = args.bins_before
    bins_after_move = args.bins_after

    trial_length = bins_before_move + bins_after_move
    n_trials = trial_data.shape[0] // trial_length

    for trial_id, trial in trial_data.groupby('trial_id'):
        trial_id_trial_info = trial_info[trial_info['trial_id'] == trial_id]

        # Get the untill movement in ms.
        move_time = (trial_id_trial_info['move_onset_time'].iloc[0] / np.timedelta64(1, 'ms')) - (trial_id_trial_info['start_time'].iloc[0] / np.timedelta64(1, 'ms'))
        # Get the number of bins until movement.
        move_bin = int(move_time // binsize)

        y_heldin_t = torch.tensor(trial.spikes.values)
        y_heldout_t = torch.tensor(trial.heldout_spikes.values)

        # Crop the trials arount the move_onset bin with offsets before and after that bin.
        y_t = torch.concat(
            [y_heldin_t[move_bin-bins_before_move:move_bin+bins_after_move, :], y_heldout_t[move_bin-bins_before_move:move_bin+bins_after_move, :]], dim=-1
        )

        y.append(y_t.reshape(1, trial_length, n_neurons))
        labels.append(torch.tensor(trial.hand_vel.values[move_bin-bins_before_move:move_bin+bins_after_move, :]).reshape(1, trial_length, 2))

    y = torch.concat(y, dim=0)
    labels = torch.concat(labels, dim=0)
    
    print(y.shape)
    print(labels.shape)
    '''

    # Align the trials arount the move_onset bin with offsets before and after that bin.
    y = []
    labels = []
    target_pos = []
    barrier_pos = []
    active_targets = []
    conds = []
    true_targets = []

    trial_length = bins_before + bins_after
    n_trials = trial_data.shape[0] // trial_length

    for trial_id, trial in trial_data.groupby('trial_id'):
        trial_id_info = trial_info[trial_info['trial_id'] == trial_id]

        # Get the untill movement in ms.
        move_time = (trial_id_info['move_onset_time'].iloc[0] / np.timedelta64(1, 'ms')) - (trial_id_info['start_time'].iloc[0] / np.timedelta64(1, 'ms'))
        # Get the number of bins until movement.
        move_bin = int(move_time // binsize)

        y_heldin_t = torch.tensor(trial.spikes.values)
        y_heldout_t = torch.tensor(trial.heldout_spikes.values)

        # Crop the trials arount the move_onset bin with offsets before and after that bin.
        y_t = torch.concat(
            [y_heldin_t[move_bin-bins_before:move_bin+bins_after, :], y_heldout_t[move_bin-bins_before:move_bin+bins_after, :]], dim=-1
        )

        y.append(y_t.reshape(1, trial_length, n_neurons))
        labels.append(torch.tensor(trial[f'{obj}_{mes}'].values[move_bin-bins_before:move_bin+bins_after, :]).reshape(1, trial_length, 2))

        target_pos.append(trial_id_info.target_pos.values[0])
        barrier_pos.append(trial_id_info.barrier_pos.values[0])
        active_targets.append(int(trial_id_info.active_target.values[0]))
        true_targets.append(trial_id_info.target_pos.values[0][int(trial_id_info.active_target.values[0])])

        for cond, trial_ids in orig_conds.items():
            if trial_id in trial_ids:
                conds.append(cond)

    y = torch.concat(y, dim=0).float()
    labels = torch.concat(labels, dim=0).float()
    conds = torch.tensor(conds)
    active_targets = torch.tensor(active_targets)
    true_targets = torch.tensor(true_targets)

    print(y.shape)
    print(labels.shape)
    print(conds.shape)
    #print(len(target_pos))
    #print(len(barrier_pos))
    #print(active_targets.shape)
    print(true_targets.shape)
    
    target_bins = []
    gocue_bins = []
    move_bins = []
    event_bins = []

    for i, _ in enumerate(trials):
        trial_id = i + n_null_trials
        trial_id_trial_info = trial_info[trial_info['trial_id'] == trial_id]

        # target : go
        delay = (((trial_id_trial_info['go_cue_time'].iloc[0] / np.timedelta64(1, 'ms')) - (trial_id_trial_info['target_on_time'].iloc[0] / np.timedelta64(1, 'ms'))) // binsize)
        # go : move
        prep = (((trial_id_trial_info['move_onset_time'].iloc[0] / np.timedelta64(1, 'ms')) - (trial_id_trial_info['go_cue_time'].iloc[0] / np.timedelta64(1, 'ms'))) // binsize)

        target_on = (((trial_id_trial_info['target_on_time'].iloc[0] / np.timedelta64(1, 'ms')) - (trial_id_trial_info['start_time'].iloc[0] / np.timedelta64(1, 'ms'))) // binsize)
        gocue = (((trial_id_trial_info['go_cue_time'].iloc[0] / np.timedelta64(1, 'ms')) - (trial_id_trial_info['start_time'].iloc[0] / np.timedelta64(1, 'ms'))) // binsize)
        move_onset = (((trial_id_trial_info['move_onset_time'].iloc[0] / np.timedelta64(1, 'ms')) - (trial_id_trial_info['start_time'].iloc[0] / np.timedelta64(1, 'ms'))) // binsize)

        target_bins.append(target_on - move_onset + bins_before)
        gocue_bins.append(gocue - move_onset + bins_before)
        move_bins.append(bins_before)

    event_bins.append(torch.tensor(target_bins))
    event_bins.append(torch.tensor(gocue_bins))
    event_bins.append(torch.tensor(move_bins))
    event_bins = torch.stack(event_bins)
    event_bins = event_bins.permute(1, 0)
    
    print(event_bins.shape)
    
    save_root_path = 'data/'

    train_data, valid_data, test_data = {}, {}, {}
    n_trials, seq_len, n_neurons = y.shape
    n_valid_trials = n_valid_trials = int(args.test_split * n_trials)

    # obs: observations
    train_data['y_obs'] = torch.Tensor(y[:-n_valid_trials])
    valid_data['y_obs'] = torch.Tensor(y[-n_valid_trials:-n_valid_trials // 2])
    test_data['y_obs'] = torch.Tensor(y[-n_valid_trials // 2:])

    # 'n_bins_enc': Number of time bins used later by in modeling for enconding (default full trial).
    # 'n_bins_obs': originaly observed trial length (after alignment)
    # Same for 'n_neurons_obs' and 'n_neurons_enc'.
    train_data['n_bins_obs'] = valid_data['n_bins_obs'] = test_data['n_bins_obs'] = seq_len
    train_data['n_bins_enc'] = valid_data['n_bins_enc'] = test_data['n_bins_enc'] = seq_len
    train_data['n_neurons_obs'] = valid_data['n_neurons_obs'] = test_data['n_neurons_obs'] = n_neurons
    train_data['n_neurons_enc'] = valid_data['n_neurons_enc'] = test_data['n_neurons_enc'] = n_neurons

    # Save a 1D array for event bins for each data split, for each trial, for each event.
    # Note: the o here in event_bins[0] is the session-animal group.
    for event_id, event in enumerate(['targrt_on_bin', 'go_cue_bin', 'move_onset_bin']):
        train_data[event] = torch.Tensor(np.array(event_bins[:-n_valid_trials, event_id]))
        valid_data[event] = torch.Tensor(np.array(event_bins[-n_valid_trials:-n_valid_trials // 2, event_id]))
        test_data[event] = torch.Tensor(np.array(event_bins[-n_valid_trials // 2:, event_id]))

    train_data[f'{obj}_{mes}'] = torch.Tensor(np.array(labels[:-n_valid_trials, :, :]))
    valid_data['hand_vel'] = torch.Tensor(np.array(labels[-n_valid_trials:-n_valid_trials // 2, :, :]))
    test_data['hand_vel'] = torch.Tensor(np.array(labels[-n_valid_trials // 2:, :, :]))
    
    train_data['conds'] = torch.Tensor(np.array(conds[:-n_valid_trials]))
    valid_data['conds'] = torch.Tensor(np.array(conds[-n_valid_trials:-n_valid_trials // 2]))
    test_data['conds'] = torch.Tensor(np.array(conds[-n_valid_trials // 2]))
    
    train_data['true_target'] = torch.Tensor(np.array(true_targets[:-n_valid_trials, :]))
    valid_data['true_target'] = torch.Tensor(np.array(true_targets[-n_valid_trials:-n_valid_trials // 2, :]))
    test_data['true_target'] = torch.Tensor(np.array(true_targets[-n_valid_trials // 2:, :]))

    '''
    for label_id, label in enumerate(label_cols):
        train_data[f'{label[0]}_{label[1]}'] = torch.Tensor(np.array(labels[:-n_valid_trials, :, :]))
        valid_data[f'{label[0]}_{label[1]}'] = torch.Tensor(np.array(labels[-n_valid_trials:-n_valid_trials // 2, :, :]))
        test_data[f'{label[0]}_{label[1]}'] = torch.Tensor(np.array(labels[-n_valid_trials // 2:, :, :]))
    '''
    torch.save(train_data, save_root_path + f'data_train_{binsize}ms.pt')
    torch.save(valid_data, save_root_path + f'data_valid_{binsize}ms.pt')
    torch.save(test_data, save_root_path + f'data_test_{binsize}ms.pt')

    print('Data splits (train/valid/test) saved into the "data" folder.')

    
if __name__ == '__main__':
    main()