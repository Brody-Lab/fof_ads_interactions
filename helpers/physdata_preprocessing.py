import numpy as np
import pandas as pd
import os

import scipy.io as sio
from scipy import stats
from sklearn.metrics import roc_auc_score
from helpers.rasters_and_psths import make_psth
import helpers.phys_helpers


def load_phys_data_from_Cell(
        filename,
        remove_violations=True,
        nboot_roc=10000):
    '''function to load matlab file Cell into python. 
    See https://github.com/Brody-Lab/npx-utils for more info about the Cells struct. 
    Args:
        filename: path to where the matlab file is located
    Returns:
        df_trial: dataframe with trial-related info, each row is a trial 
        df_cell: dataframe with cell-related info, each row is a cell
        data_prms: dict other accessory information about data collection and preprocessing
    '''
    # load the Trial struct containing all the behavior-related information
    behavior = sio.loadmat(
        filename,
        simplify_cells=True,
        variable_names="Trials")

    # removing field which either aren't loaded in python properly or have weirdly inconsistent sizes
    [behavior['Trials']['laser'].pop(x) for x in ['triggerEvent',
                                                  'brain_area',
                                                  'laterality',
                                                  'stim_per']]
    [behavior['Trials']['pharma'].pop(x) for x in ['manip']]
    [behavior['Trials']['stateTimes'].pop(x) for x in ['right_clicks',
                                                       'right_click_times',
                                                       'right_click_trial',
                                                       'left_clicks',
                                                       'left_click_times',
                                                       'left_click_trial',
                                                       'shuffled_right_clicks',
                                                       'shuffled_left_clicks',
                                                       'shuffled_right_click_times',
                                                       'shuffled_left_click_times']]

    # now concatenating to make the final dataframe
    df_trial = pd.concat([pd.DataFrame(behavior['Trials']['laser']),
                          pd.DataFrame(behavior['Trials']['pharma']),
                          pd.DataFrame(behavior['Trials']['stateTimes'])], axis=1)
    [behavior['Trials'].pop(x) for x in ['rat',
                                         'sess_date',
                                         'laser',
                                         'pharma',
                                         'stateTimes']]
    df_trial = pd.concat([pd.DataFrame(behavior['Trials']), df_trial], axis=1)

    # adding columns in which click times are wrt cpoke_in time
    df_trial['stereo_click'] = df_trial['clicks_on']
    bupVars = ['leftBups', 'rightBups']
    for i in range(len(df_trial)):
        for v in bupVars:
            if isinstance(df_trial.loc[i, v], float):
                df_trial.at[i, v] = np.array([], dtype=np.uint8)
            elif len(df_trial.loc[i, v]) > 0:
                df_trial.at[i, v] = df_trial.loc[i, v][1:] - \
                    df_trial.loc[i, v][0] + \
                    df_trial.loc[i, 'clicks_on']

    # add some other relevant fields to df_trial
    if remove_violations is True:
        df_trial = remove_violations_from_df_trial(df_trial)
    df_trial = add_hit_pokedR_to_df_trial(df_trial)

#   load other params from the data file - these will stay as a dict
    params = [
        'rec',
        'jrc_file',
        'hemisphere',
        'made_by',
        'mat_file_name',
        'nTrials',
        'penetration',
        'params',
        'probe_serial',
        'rat',
        'sess_date',
        'sessid',
        'waveformSim',
        'kSpikeWindowS']
    data_prms = sio.loadmat(
        filename, simplify_cells=True, variable_names=params)
    [data_prms.pop(x) for x in ['__header__', '__version__', '__globals__']]

    # now load each cell related information and store it is as a dataframe
    params = [
        'AP',
        'ML',
        'DV',
        'bank',
        'distance_from_tip',
        'electrode',
        'waveform',
        'raw_spike_time_s',
        'frac_isi_violation',
        'clusterNotes',
        'unitVppRaw',
        'unitCount',
        'unitISIRatio',
        'unitIsoDist',
        'unitLRatio',
        'firing_rate',
        'regions',
        'ks_good']
    cell_data = sio.loadmat(
        filename, simplify_cells=True, variable_names=params)
    if len(cell_data['waveform']) != len(cell_data['unitCount']):
        cell_data.pop('waveform')
    cell_data['spiketime_s'] = cell_data.pop('raw_spike_time_s')

    # assigning each cell its region based on penetration dict
    region_list = [data_prms['penetration']['regions'][i]['name']
                   for i in range(len(data_prms['penetration']['regions']))]
    cell_data['region'] = [str(region_list[r - 1])
                           for r in cell_data['regions']]
    [cell_data.pop(x)
     for x in ['__header__', '__version__', '__globals__', 'regions']]
    df_cell = pd.DataFrame(cell_data)
    df_cell['hemisphere'] = data_prms['penetration']['hemisphere']

    # assign unique cell IDs - this has to follow creation of df_cell
    # so that all recorded cells are accounted for and get a unique id
    ratname = data_prms['rec']['rat']
    sessdate = data_prms['rec']['sessiondate'].replace('-', '')
    df_cell['cell_ID'] = ''
    for i in range(len(df_cell)):
        df_cell.loc[i, 'cell_ID'] = ratname + '_' + sessdate + '_' + str(i)
    df_cell.set_index(df_cell.columns[-1], inplace=True)
    df_cell.reset_index(inplace=True)

    # add side sel related info
    df_cell = populate_Cells_with_statistics(
        filename, df_cell, df_trial, nboot=nboot_roc)

    # reset region names, just get the regions I want and arrange them by DV
    df_cell['region'] = df_cell['region'].replace(
        {'DMS': 'ADS', "['M2' 'FOF']": 'FOF', 'M2': 'FOF'})
    df_cell = df_cell[df_cell['region'].str.match("ADS") |
                      df_cell['region'].str.match("FOF")].reset_index(drop=True)
    df_cell = df_cell.sort_values(by=['DV', 'cell_ID']).reset_index(drop=True)

    return df_trial, df_cell, data_prms


def load_PETH_from_Cell(filename):
    spikedata = sio.loadmat(filename, simplify_cells=True,
                            variable_names=['recorded', 'spike_time_s'])
    [spikedata.pop(x) for x in ['__header__', '__version__', '__globals__']]

    df_spktime = dict()
    # add each cell's spiking data for each trial (each alignment is added as a new column)
    spike_keys = spikedata['spike_time_s'].keys()
    for c in range(len(spikedata['spike_time_s']['cpoke_in'])):
        colname = 'recorded_cell_' + str(c)
        df_spktime[colname] = pd.Series(spikedata['recorded'][c])
        for k in spike_keys:
            colname = k + '_cell_' + str(c)
            df_spktime[colname] = pd.Series(spikedata['spike_time_s'][k][c])


def add_hit_pokedR_to_df_trial(df_trial):
    df_trial['hit_pokedR'] = 2*df_trial['pokedR'] + 0.5*df_trial['is_hit']
    df_trial['prev_trial'] = np.concatenate(
        ([2.5], 2*df_trial['pokedR'][:-1] + 0.5*df_trial['is_hit'][:-1]))

    # variables = ['hit_pokedR', 'prev_trial']
    # for var in variables:
    #     df_trial.loc[df_trial[var] == 2, var] = 'right error'
    #     df_trial.loc[df_trial[var] == 0, var] = 'left error'
    #     df_trial.loc[df_trial[var] == 2.5, var] = 'right correct'
    #     df_trial.loc[df_trial[var] == 0.5, var] = 'left correct'
    # Mapping numerical values to corresponding labels
    mapping = {2: 'right error', 0: 'left error', 2.5: 'right correct', 0.5: 'left correct'}
    for var in ['hit_pokedR', 'prev_trial']:
        df_trial[var] = df_trial[var].map(mapping)

    # 1 if the bias due to trial history should be to right
    # (assuming win stay lose switch policy)
    df_trial['prev_bias'] = np.where(np.any([df_trial['prev_trial'] == 'right correct',
                                             df_trial['prev_trial'] == 'left error'], axis=0),
                                     True, False)

    df_trial['prev_bias_right'] = np.where(df_trial['prev_bias'], True, False)
    df_trial['prev_bias_left'] = np.where(df_trial['prev_bias'], False, True)
    return df_trial


def remove_violations_from_df_trial(df_trial):
    df_trial = df_trial[(df_trial['trial_type'] == 'a')]
    df_trial = df_trial[(df_trial['violated'] == 0)].reset_index()
    return df_trial


def shuffle_roc_auc_score(labels, scores):
    np.random.shuffle(scores)
    return roc_auc_score(labels, scores)


def populate_Cells_with_statistics(filename, df_cell, df_trial, nboot=10000):

    np.random.seed(1)
    filename = filename.replace('Cells', 'Cells_sidesel')
    filename = filename.replace('.mat', '.npy')

    if os.path.exists(filename):
        side_dict = np.load(filename, allow_pickle=True).item()
    else:
        print("\nRunning bootstraps to compute side selectivity\n")

        # do bootstrapping parallely
        import multiprocessing
        from joblib import Parallel, delayed
        num_cores = multiprocessing.cpu_count()

        window = [0, 1000]
        align_to = 'clicks_on'
        split_by = 'pokedR'
        pre_mask = None
        post_mask = "clicks_off"
        filter_type = 'gaussian'
        filter_width = 75
        binsize = 25

        side_dict = dict()
        side_dict['cell_ID'] = []
        side_dict['auc'] = []
        side_dict['side_pref'] = []
        side_dict['side_pref_p'] = []
        side_dict['stim_fr'] = []

#         df_trial["clicks_off-100"] = df_trial['clicks_off'] - 0.1

        for i in range(len(df_cell)):

            # compute firing rate during clicks
            stim_fr = np.nanmean(make_psth(
                df_cell['spiketime_s'][i],
                df_trial,
                plot=False,
                window=[0, 1000],
                binsize=binsize,
                pre_mask=None,
                align_to='clicks_on',
                post_mask='clicks_off',
                filter_type=None,
                split_by=None)['data'][0]['mean'])

            this_psth = make_psth(
                df_cell['spiketime_s'][i],
                df_trial,
                plot=False,
                window=window,
                binsize=binsize,
                pre_mask=pre_mask,
                post_mask=post_mask,
                filter_type=filter_type,
                filter_w=filter_width,
                align_to=align_to,
                split_by=split_by)

            A = this_psth['trial_fr'][0]
            B = this_psth['trial_fr'][1]

            A = A[~np.isnan(A)]
            B = B[~np.isnan(B)]

            labels = np.hstack((np.zeros(len(A)), np.ones(len(B))))
            scores = np.hstack((A, B))
            data_auc = roc_auc_score(labels, scores)
            boot_auc = Parallel(n_jobs=num_cores)(
                delayed(shuffle_roc_auc_score)(labels, scores) for k in range(nboot))

            side_pref = int(data_auc > 0.5)
            if side_pref == 1:
                side_pref_p = np.mean(boot_auc >= data_auc)
            else:
                side_pref_p = np.mean(boot_auc <= data_auc)

            side_dict['cell_ID'].append(df_cell.loc[i, 'cell_ID'])
            side_dict['auc'].append(data_auc)
            side_dict['side_pref'].append(side_pref)
            side_dict['side_pref_p'].append(side_pref_p)
            side_dict['stim_fr'].append(stim_fr)

        dirpath = os.path.split(filename)[0]
        os.makedirs(dirpath, exist_ok=True)
        np.save(filename, side_dict)

    return pd.merge(df_cell, pd.DataFrame(side_dict), on='cell_ID')


# def populate_Cells_with_statistics(df_cell, df_trial):
#     window = [0, 900]
#     align_to = 'clicks_on'
#     split_by = 'pokedR'
#     pre_mask = None
#     post_mask = "clicks_off"
#     binsize = 25
#     filter_type = 'gaussian'
#     filter_w = 150

#     side_pref = []
#     side_pref_p = []
#     stim_fr = []

#     for i in range(len(df_cell)):

#         this_psth = make_psth(df_cell['spiketime_s'][i],
#                               df_trial[df_trial['violated']==0].reset_index(),
#                               plot = False,
#                               window = window,
#                               binsize = binsize,
#                               pre_mask = pre_mask,
#                               post_mask = post_mask,
#                               align_to = align_to,
#                               split_by = split_by,
#                               filter_type = filter_type,
#                               filter_w = filter_w)
#         mean_fr = []
#         masked_data = []
#         for cond in np.sort(list(this_psth['data'].keys())):
#             mean_fr.append(np.nanmean(this_psth['data'][cond]['mean']))
#             masked_data.append(apply_mask(this_psth['trial_fr'][cond],
#                                            this_psth['mask'][cond]).ravel())

#         # remove nans
#         masked_data[0] = [x for x in masked_data[0] if np.isnan(x) == False]
#         masked_data[1] = [x for x in masked_data[1] if np.isnan(x) == False]

#         if mean_fr[1] == mean_fr[0] == 0:
#             side_pref_p.append(1)
#             side_pref.append(np.nan)
#         else:
#             stat, pvalue = stats.mannwhitneyu(masked_data[0], masked_data[1])
#             side_pref_p.append(pvalue)
#             if mean_fr[1] > mean_fr[0]:
#                 side_pref.append('r')
#             else:
#                 side_pref.append('l')
#         stim_fr.append(np.nanmean(mean_fr))

#     df_cell['side_pref'] = side_pref
#     df_cell['side_pref_p'] = side_pref_p
#     df_cell['stim_fr'] = stim_fr
