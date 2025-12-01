import numpy as np
import pandas as pd
import random, os, sys
import datetime

basedir = '/home/dikshag/pbups_ephys_analysis/pbups_phys/multiregion_RNN/'
sys.path.append(basedir)
sys.path.append(basedir + 'base/')
sys.path.append(basedir + 'analysis/')
from rnn import *
from simulations import *
from task import *
from plotting_general import *


def get_base_path():
    
    if os.path.isdir('/home/dikshag/pbups_ephys_analysis/saved_results/rnn_runs/'):
        base_path = '/home/dikshag/pbups_ephys_analysis/saved_results/rnn_runs/'
    else:
        base_path = '/Users/dikshagupta/ondrive/analysisDG/PBups_Phys/saved_results/FOF_ADS_rnn/rnn_runs/'
        
    return base_path


def get_figure_dir():
    
    if os.path.isdir('/home/dikshag/pbups_ephys_analysis/saved_results/manuscript_figures/'):
        figure_dir = '/home/dikshag/pbups_ephys_analysis/saved_results/manuscript_figures/'
    else:
        figure_dir = '/Users/dikshagupta/ondrive/analysisDG/PBups_Phys/saved_results/FOF_ADS_rnn/rnn_runs/'
        
    return figure_dir
    
    

def set_rnn_save_path(run_name = None):
    
    base_path = get_base_path()
    if run_name == None:
        run_name = 'run_' + datetime.datetime.now().strftime("%y%m%d_%H%M%S")+'/'
    else:
        run_name = 'run_' + run_name + chr(random.randrange(97, 97 + 26))+ datetime.datetime.now().strftime("%y%m%d_%H%M%S")+'/'
    dirname = base_path + run_name
    os.mkdir(dirname)
    return dirname 




def save_run(mdl, network_params, train_params):
    dirname = train_params['training_weights_path']
    np.save(dirname + 'network_params', network_params)
    mdl.save(dirname + 'final_weights')
    
    
    
def savethisfig(dirname, name):
    '''saves the current figure as pdf with the given name. 
    Prints the location at whcih figure was saved.
    '''
    # dirname = os.path.split(mdl.load_weights_path)[0] + os.sep
    filename = dirname + name + ".pdf"
    plt.savefig(filename, 
            dpi=300, 
            format='pdf', 
            bbox_inches='tight',
            transparent = "true")
    print("YESS QUEEN: file saved! \nJust look over here: " + filename )


       
def make_connectivity_matrices(network_params):
    
    N_rec = network_params['N_rec']
    N_in = network_params['N_in']
    N_out = network_params['N_out']
    k = network_params['N_rec_multiply']
    
    # assert(N_rec == 300)
    assert(N_out == 1)
    assert(network_params['dale_ratio'] == 0.5)
    

    # output is read out from separate groups of excitatory neurons
    output_connectivity = np.ones((N_out, N_rec))

    # input goes only to FOF and ADS
    input_connectivity = np.zeros((N_rec, N_in))
    input_connectivity[:k*80, :] = 1
    input_connectivity[k*200:, :] = 1

    # constructing recurrent matrix to give a semblance of areas
    rec_connectivity = np.ones((N_rec, N_rec))
    rec_connectivity[k*80:k*180, :k*80] = 0
    rec_connectivity[k*200:, k*80:k*180] = 0
    rec_connectivity[k*80:k*180,k*180:k*200] = 0
    rec_connectivity[k*200:, k*180:k*200] = 0
    rec_connectivity[:k*80, k*200:] = 0
    rec_connectivity[k*180:k*200, k*200:] = 0
    rec_connectivity[k*250:k*300, k*200:k*250] = 0
    rec_connectivity[k*200:k*250, k*250:k*300] = 0


    rec_connectivity[k*80:k*115, k*250:k*300] = 0
    rec_connectivity[k*150:k*165, k*250:k*300] = 0
    rec_connectivity[k*115:k*150, k*200:k*250] = 0
    rec_connectivity[k*165:k*180, k*200:k*250] = 0

    # FOF inhibition limited to within hemisphere
    rec_connectivity[k*190:k*200, k*180:k*190] = 0
    rec_connectivity[k*180:k*190, k*190:k*200] = 0
    rec_connectivity[k*180:k*190, k*40:k*80] = 0
    rec_connectivity[k*190:k*200, k*0:k*40] = 0
    rec_connectivity[k*40:k*80, k*180:k*190] = 0
    rec_connectivity[:k*40, k*190:k*200] = 0

    #     # # sparsify across FOF connectivity
    #     rec_connectivity[40:80, 20:40] = 0
    #     rec_connectivity[:40, 60:80] = 0
    #     rec_connectivity[190:200, 0:40] = 0
    #     rec_connectivity[:40, 190:200] = 0
    #     rec_connectivity[40:80, 180:190] = 0
    #     rec_connectivity[180:190, 40:80] = 0
    #     rec_connectivity[200:, 25:40] = 0
    #     rec_connectivity[200:, 65:80] = 0
    #     rec_connectivity[250:, 0:20] = 0
    #     rec_connectivity[200:250, 40:60] = 0


    # March 31: sparsify across FOF connectivity
    rec_connectivity[k*40:k*80, k*20:k*40] = 0
    rec_connectivity[:k*40, k*60:k*80] = 0
    rec_connectivity[k*200:k*250, k*30:k*40] = 0
    rec_connectivity[k*250:k*300, k*20:k*40] = 0
    rec_connectivity[0:k*40, k*60:k*80] = 0
    rec_connectivity[k*200:k*250, k*60:k*80] = 0
    rec_connectivity[k*250:k*300, k*70:k*80] = 0


    network_params['input_connectivity'] = input_connectivity
    network_params['rec_connectivity'] = rec_connectivity
    network_params['output_connectivity'] = output_connectivity

    
    return network_params





def format_data(x,y,m,params,output,activity):

    # generate df_trial
    ntrials = np.shape(x)[0]
    df_trial = pd.DataFrame(columns = params[0].keys())
    for i in range(len(params)):
        df_trial = pd.concat([df_trial, pd.DataFrame.from_records([params[i]])], ignore_index = True)
    # df_trial['choice'] = [np.random.rand() <= expit(10*np.diff(np.mean(output[i,150:,:], axis = 0)))[0] for i in range(ntrials)]

    df_trial['choice'] = [np.mean(output[i,np.squeeze(m[i]) == 1,:]) > 0 for i in range(ntrials)]
    
    # not assigning randomly for 0 delta clicks
    df_trial['correct'] = 1*(df_trial['Δclicks'] >= 0)
    df_trial.choice_target_end = df_trial.choice_target_end.apply(pd.to_numeric) 
    df_trial.choice = df_trial.choice.apply(pd.to_numeric) 
    df_trial['history_bias'] = df_trial['history_bias'].astype(float)
    df_trial['difficulty'] = pd.cut(df_trial['gamma'], 
                                    bins = 4, 
                                    labels = ["L easy", "L hard", "R hard", "R easy"])
    
    # make it neuron x trial x time
    activity = np.array(activity).transpose(2,0,1) 
    
    k = int(activity.shape[0]/300)
    
    # put all the FOF neurons together
    permutation = [np.arange(k*80), np.arange(k*180,k*200), np.arange(k*80,k*180), np.arange(k*200,k*300)]
    permutation = [item for sublist in permutation for item in sublist]
    activity = activity[permutation,:,:]
    
    return df_trial, activity,  {'x': x, 'y': y, 'm': m, 'output': output}





   
def make_perturbation_inputs_mul(N_steps, perturb_type, perturb_group, dt = 10, gain = 0):

    # this function currently assumes that this is for a 300x300 network
    # with first 80 and 180:200 neurons as FOF excitatory/inhibitory neurons
    # 80:180 "in between" neurons
    # and 200:300 ADS neurons. First half of each side is considered 
    t_connectivity = np.ones((300,300))
    t_connectivity_perturb = np.ones((300,300))
 
    
    which_neurons = {'left_FOF': np.ix_(range(300), range(40)),
                 'right_FOF': np.ix_(range(300), range(40,80)),
                 'bi_FOF': np.ix_(range(300), range(80)),
                 'left_ADS': np.ix_(range(300), range(200,250)),
                 'right_ADS': np.ix_(range(300), range(250,300)),
                 'bi_ADS': np.ix_(range(300), range(200,300)),
                 'left_proj': np.ix_(range(200,250), range(80)),
                 'right_proj': np.ix_(range(250,300), range(80)),
                 'left_ADSproj': np.ix_([*range(0,40), *range(180,190)],
                                        range(80,180)),
                 'right_ADSproj': np.ix_([*range(40,80), *range(190,200)],
                                         range(80,180)),
                 'bi_ADSproj': np.ix_([*range(0,80), *range(180,200)],
                                       range(80, 180)),
                 'leftFOF_bothADS': np.ix_(range(200,300), range(40)),
                 'rightFOF_bothADS': np.ix_(range(200,300), range(40,80)),
                 'leftFOF_leftADS': np.ix_(range(200,250), range(40)),
                 'leftFOF_rightADS': np.ix_(range(250,300), range(40)),
                 'rightFOF_leftADS': np.ix_(range(200,250), range(40,80)),
                 'rightFOF_rightADS': np.ix_(range(250,300), range(40,80)),
                 'leftFOF_rec': np.ix_(range(40), range(40)),
                 'rightFOF_rec': np.ix_(range(40,80), range(40,80)),
                 'leftFOF_leftADS_rec': np.ix_([*range(200,250), *range(40)], range(40)),
                 'rightFOF_rightADS_rec': np.ix_([*range(250,300), *range(40,80)], range(40,80)),
                 'leftFOF_leftADS_bi': np.ix_([*range(200,250), *range(40,80)], range(40)),
                 'rightFOF_rightADS_bi': np.ix_([*range(250,300), *range(40)],range(40,80)),
                 'leftFOF_bi': np.ix_(range(40,80), range(40)),
                 'rightFOF_bi': np.ix_(range(40), range(40,80)),
                 'leftFOF_recbi': np.ix_(range(80), range(40)),
                 'rightFOF_recbi': np.ix_(range(80), range(40,80))} 
    
    if perturb_type == 'whole_trial':
        t0 = 0
        t1 = N_steps
    elif perturb_type == 'first_half':
        t0 = int(500/dt)
        t1 = int(1000/dt)
    elif perturb_type == 'second_half':
        t0 = int(1000/dt)
        t1 = int(1500/dt)
    else:
        print('UNDEFINEED INACTIVATION TYOPE')
        return
    
    t_connectivity_perturb[which_neurons[perturb_group]] = gain    
    t_connectivity= np.repeat(t_connectivity[np.newaxis, :, :], N_steps, axis = 0)
    t_connectivity[t0:t1] = t_connectivity_perturb
    
    return t_connectivity
    
    
    
    
def make_perturbation_inputs_add(N_steps, perturb_type, perturb_group, dt, magnitude = -0.25, k = 1):
    
    
    
    which_neurons = {'left_FOF': range(k*40),
                   'right_FOF': range(k*40, k*80),
                   'bi_FOF': range(k*80),
                   'left_ADS': range(k*200,k*250),
                   'right_ADS': range(k*250,k*300),
                   'bi_ADS': range(k*200, k*300),
                   'left_proj': range(k*80),
                   'right_proj': range(k*80)}
    
    if perturb_type == 'whole_trial':
        t0 = 0
        t1 = N_steps
    elif perturb_type == 'first_half':
        t0 = int(500/dt)
        t1 = int(1000/dt)
    elif perturb_type == 'second_half':
        t0 = int(1000/dt)
        t1 = int(1500/dt)
    else:
        print('UNDEFINEED INACTIVATION TYOPE')
        return
    
    opto_inputs = np.zeros((N_steps, k*300))
    opto_inputs[t0:t1, which_neurons[perturb_group]] = magnitude

    use_opto_mask = 1 if 'proj' in perturb_group else 0
    if perturb_group == 'left_proj':
        opto_mask = np.zeros((k*300, k*300))
        opto_mask[np.ix_(range(k*200,k*250), range(k*80))] = 1
    elif perturb_group == 'right_proj':
        opto_mask = np.zeros((k*300, k*300))
        opto_mask[np.ix_(range(k*250,k*300), range(k*80))] = 1        
    else:
        opto_mask = np.zeros((k*300, k*300))
    
    return opto_inputs, opto_mask, use_opto_mask
    
    
    
    
def reinitialize_network(file, new_params = None):
    
    base_path = get_base_path()
    dirname = base_path + file + os.sep
    network_params = np.load(dirname + 'network_params.npy', allow_pickle = True).item()
    network_params['load_weights_path'] = dirname + 'final_weights.npz'
    
    if new_params is not None:
        for k in list(new_params):
            network_params[k] = new_params[k]
    FOF_ADS = Basic(network_params)

    # and the task class
    task_keys = list(PoissonClicks().__dict__.keys())
    task_keys = [ele for ele in task_keys if ele not in ['alpha', 'N_steps']]
    prm_dict = {key:network_params[key] for key in task_keys}
    pc_data = PoissonClicks(**prm_dict)

    model_sim =  BasicSimulator(rnn_model = FOF_ADS)
    
    return FOF_ADS, pc_data, model_sim
    
    
    
    
def compute_fracR(df_trial):
    Δcuts, bins = pd.cut(df_trial['Δclicks'], bins = 20, retbins = True)
    bin_centers = bins[:-1] + np.diff(bins)[0]/2
    fracR = df_trial.groupby(Δcuts)['choice'].mean()    
    
    return fracR

    
    
# def reinitialize_network(dirname, new_params = None):
    
#     network_params = np.load(dirname + 'network_params.npy', allow_pickle = True).item()
#     network_params['load_weights_path'] = dirname + 'final_weights.npz'
    
#     if new_params is not None:
#         for k in list(new_params):
#             network_params[k] = new_params[k]
#     FOF_ADS = Basic(network_params)
    
#     # and the task class
#     task_keys = list(PoissonClicks().__dict__.keys())
#     task_keys = [ele for ele in task_keys if ele not in ['alpha', 'N_steps', 'N_rec_multiply']]
#     prm_dict = {key:network_params[key] for key in task_keys}
#     pc_data = PoissonClicks(**prm_dict)
    
#     model_sim = BasicSimulator(rnn_model = FOF_ADS)
    
#     return FOF_ADS, pc_data, model_sim