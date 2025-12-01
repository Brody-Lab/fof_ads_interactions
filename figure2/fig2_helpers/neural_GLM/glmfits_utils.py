import sys
import numpy as np
from collections import OrderedDict
from basis_kernels import make_kernel

sys.path.insert(1, '../../../figure_code/')
from my_imports import *
from helpers.rasters_and_psths import make_psth, adjust_window
plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.Spectral(np.linspace(0,1,8)))
plt.rcParams["font.family"] = 'Sans-serif'

def get_Y(df_cell, df_trial, p, region = None):
    
    align_to = p['align_to']
    window = p['window']
    pre_mask = p['pre_mask']
    post_mask = p['post_mask']
    
    if region is not None:
        mask = 'region == "' + region + '"'
    else:
        mask = 'region == "FOF" or region =="ADS" '
    
    Y = []
    for i in range(len(df_cell.query(mask))):
        fr = make_psth(
            df_cell.query(mask).iloc[i]['spiketime_s'],
            df_trial,
            align_to=align_to,
            window=window,
            binsize=p['binsize'],
            pre_mask=pre_mask,
            post_mask=post_mask,
            filter_w=p['filter_w'],
            filter_type=p['filter_type'],
            plot=False)

        Y.append(fr['trial_fr'][0])

    Y = np.array(Y)
    Y = np.moveaxis(Y, [0,1,2], [2,0,1]) * p['binsize']/1000
    
    return Y
 



def get_X(df_trial, p):
    
    this_window = [0, 3000] # bigger window to make sure choice gets in 
    binsize_s = p['binsize']*1e-3
    window_s = [w*1e-3 for w in adjust_window(this_window , p['binsize'])]
    edges = np.arange(window_s[0], window_s[1] + binsize_s, binsize_s)

    X = np.zeros((len(df_trial),len(edges)-1, len(p['covariates'])))
    
    def compute_histogram(
        row, 
        edges, 
        column, 
        align_to_column, 
        conditional_column = None, 
        conditional_value = None):
        
        if conditional_column is not None:
            if row[conditional_column] != conditional_value:
                return np.zeros(len(edges) - 1)  # Return zeros if the condition is not met
        differences = row[column] - row[align_to_column]
        histogram, _ = np.histogram(differences, bins=edges)
        return histogram
    
    for i, covar in enumerate(p['covariates']):
        
        if covar in ['leftBups', 'rightBups', 'stereo_click']:
            additional_args = (edges, covar, p['align_to'])
            
        elif covar in ['choiceL', 'choiceR']:
            cond = 1 if covar == 'choiceR' else 0
            additional_args = (edges, 'cpoke_out', p['align_to'], 'pokedR', cond)
            
        elif covar in ['cpoke_in_L', 'cpoke_in_R']:
            cond = 1 if covar == 'cpoke_in_R' else 0
            additional_args = (edges, 'cpoke_in', p['align_to'], 'pokedR', cond)
            
        elif covar in ['cpoke_in']:
            additional_args = (edges, covar, p['align_to'])
            
        else:
            raise ValueError("stumbled on an unknown covariate: {}".format(covar))
            
        
        X_values = df_trial.apply(compute_histogram, args=additional_args, axis=1)
        X[:,:,i] = np.vstack(X_values.to_numpy())
        
    return X
    
    
def get_covar_dict(p):
        
    covar_dict = OrderedDict()
    
    for covar in p['covariates']:
        
        if covar in ['leftBups', 'rightBups']:
            covar_dict[covar] = {
                'num': 12,
                'duration': 0.75,
                'peak_range': [0.01, 0.8],
                'log_scaling': True,
                'log_offset': 0.1,
                'time_offset': 0.  # <0 is acausal
                }
            
        elif covar in ['cpoke_in_L', 'cpoke_in_R', 'cpoke_in']:
            covar_dict[covar] = {
                'num': 18,
                'duration': 2.0,
                'peak_range': [0.01, 1.99],
                'log_scaling': False,
                'log_offset': 0.,
                'time_offset': 0.
                }
            
        elif covar in ['stereo_click']:
            covar_dict[covar] = {
                'num': 7,
                'duration': 0.9,
                'peak_range': [-0.05, 0.88],
                'log_scaling': False,
                'log_offset': -0.1,
                'time_offset': 0.
                }
            
        elif covar in ['choiceL', 'choiceR']:
            covar_dict[covar] = {
                'num': 20,
                'duration': 4.5,
                'peak_range': [0.1, 2.2],
                'log_scaling': False,
                'log_offset': -0.1,
                'time_offset': -2.0
                }
            
        else:
            raise ValueError("stumbled on an unknown covariate: {}".format(covar))
        
        covar_dict[covar]['kernels'] = make_kernel(covar_dict[covar], p['binsize']/1000)
        
    return covar_dict
    
    
