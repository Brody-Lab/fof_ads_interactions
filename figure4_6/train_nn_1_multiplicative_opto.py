from multiregion_RNN_utils import *
import re, sys, os

# SET RANDOM SEED
seed = np.random.randint(10**8)
tf.compat.v2.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

print(sys.argv)

if len(sys.argv) > 2:
    
    print(sys.argv[1])
    filenum = int(sys.argv[1])
    
    base_path = get_base_path()
    files = sorted([fn for fn in os.listdir(base_path) if re.findall("run_new_opto_1d_25_20_2_", fn) ])
    dirname = base_path + files[filenum] + os.sep
    print(dirname)
    
    train_params = np.load(dirname + 'train_params.npy', allow_pickle = True).item()
    network_params = np.load(dirname + 'network_params.npy', allow_pickle = True).item()
    network_params['load_weights_path'] = dirname + 'final_weights.npz'
    network_params['frac_opto'] = 0.20

    if train_params['train_withopto'] == True:
        FOF_ADS = BasicOpto(network_params)
    else:
        FOF_ADS = Basic(network_params)
        
    task_keys = list(PoissonClicks().__dict__.keys())
    task_keys = [ele for ele in task_keys if ele not in ['alpha', 'N_steps']]
    prm_dict = {key:network_params[key] for key in task_keys}
    pc = PoissonClicks(**prm_dict)
    
    dirname = base_path + files[filenum] + '/train_iter_2/'
    train_params['training_weights_path'] = dirname
    train_params['learning_rate'] = .00005 # Sets learning rate if use default optimizer Default: .001

    os.makedirs(dirname, exist_ok=True)


else:

    # INITIALIZE DIRECTORY FOR SAVING
    run_name = 'optomul2023_1d_35_100_10_'
    dirname = set_rnn_save_path(run_name = run_name)

    # INITIALIZE TASK TRIAL CLASS
    pc = PClicks_optomul(dt = 10, tau = 100, T = 1700, 
                       N_batch = 64, 
                       frac_opto = 0.25,
                       acc_bound = 8.,
                       acc_sigma_sens = 2.,
                       acc_lapse = 0.,
                       history_mod = None)


    # INITIALIZE NETWORK
    network_params = pc.get_task_params()
    network_params['name'] = 'FOF_ADS'
    N_rec = network_params['N_rec'] = 300
    network_params['b_rec_train'] = False
    network_params['W_out_train'] = True
    network_params['b_out_train'] = False

    network_params['transfer_function'] = 'relu'
    network_params['rec_noise'] = 0.15
    network_params['random_seed'] = seed

    network_params['L1_in'] = 0.
    network_params['L1_rec'] = 0.
    network_params['L1_out'] = 0.
    network_params['L2_in'] = 0.
    network_params['L2_rec'] = 0.0
    network_params['L2_out'] = 0.00005
    network_params['L2_firing_rate'] = 0.
    network_params['custom_regularization'] = None

    network_params['autapses'] = False
    dale_ratio = network_params['dale_ratio'] = 0.5
    network_params = make_connectivity_matrices(network_params)

    FOF_ADS = Basic(network_params)
    weights = FOF_ADS.get_weights()
    rec_connectivity = network_params['rec_connectivity']
    dale_vec = np.ones(N_rec)
    dale_vec[int(dale_ratio * N_rec):] = dale_ratio/(1-dale_ratio)
    dale_rec = np.diag(dale_vec) / np.linalg.norm(np.matmul(rec_connectivity,
                                                            np.diag(dale_vec)), 
                                                  axis=1)[:,np.newaxis]
    weights['W_rec'] = np.matmul(weights['W_rec'], dale_rec)

    # set the desired spectral radius
    weights['W_rec'] = weights['W_rec']*1.5/np.max(abs(np.linalg.eigvals(weights['W_rec'])))


    # make the new network with correct initialization
    network_params.update(weights)
    FOF_ADS.destruct()
    if network_params['frac_opto'] > 0:
        FOF_ADS = Basic_optomul(network_params)
    else:
        FOF_ADS = Basic(network_params)

    # SET TRAINING PARAMETERS
    train_params = {}
    train_params['save_weights_path'] = None  # Where to save the model after training. Default: None
    train_params['training_iters'] = 1200000 # number of iterations to train for Default: 50000
    train_params['learning_rate'] = .0001 # Sets learning rate if use default optimizer Default: .001
    train_params['loss_epoch'] = 50 # Compute and record loss every 'loss_epoch' epochs. Default: 10
    train_params['verbosity'] = True # If true, prints information as training progresses. Default: True
    train_params['save_training_weights_epoch'] = 2000 # save training weights every 'save_training_weights_epoch' epochs. Default: 100
    train_params['training_weights_path'] = dirname # where to save training weights as training progresses. Default: None
    train_params['optimizer'] = tf.compat.v1.train.AdamOptimizer(learning_rate=train_params['learning_rate']) # What optimizer to use to compute gradients. Default: tf.train.AdamOptimizer(learning_rate=train_params['learning_rate'])
    train_params['clip_grads'] = True # If true, clip gradients by norm 1. Default: True
    train_params['train_withopto'] = True if network_params['frac_opto'] > 0 else False

    
np.save(dirname + 'train_params', train_params)
FOF_ADS.save(dirname + 'initial_weights')

# TRAIN NETWORK
losses = FOF_ADS.train(pc, train_params)

network_params['loss_trajectory'] = losses
save_run(FOF_ADS, network_params, train_params)
FOF_ADS.destruct()





# REINITIALIZE TO PLOT MODEL PERFORMANCE
new_params = {'N_batch': 2000,
              'p_probe': 0.,
              'frac_opto': 0.,
              'probe_duration': 0}

# reinitialize the network
network_params = np.load(dirname + 'network_params.npy', allow_pickle = True).item()
network_params['load_weights_path'] = dirname + 'final_weights.npz'
for k in list(new_params):
    network_params[k] = new_params[k]
FOF_ADS = Basic(network_params)

# and task instance
task_keys = list(PoissonClicks().__dict__.keys())
task_keys = [ele for ele in task_keys if ele not in ['alpha', 'N_steps']]
prm_dict = {key:network_params[key] for key in task_keys}
pc_data = PoissonClicks(**prm_dict)

# simulate data
x, y, m, params = pc_data.get_trial_batch()
output, activity = FOF_ADS.test(x)
df_trial, activity, data_dict = format_data(x,y,m, params, output, activity)


# okay make the first major figure
fig_perf = plt.figure(constrained_layout = True, figsize = (11,11))
subfigs = fig_perf.subfigures(nrows = 2, ncols = 1, height_ratios = [1,2] )

axs0 = subfigs[0].subplots(1,4, subplot_kw=dict(box_aspect=1))
 
# plot loss trajectory
axs0[0].plot(network_params['loss_trajectory'][1:], 'grey')
axs0[0].set_ylabel("Loss")
axs0[0].set_xlabel("Training Iteration")
axs0[0].set_title("Loss During Training")

# plot final weight matrix
plot_weights(FOF_ADS.get_weights()['W_rec'], ax = axs0[1], title = "Final weight matrix")

# plot psychometric fit
plot_psych(df_trial, 'choice_target_end', axs0[2], legend = 'Target', color = 'k')
plot_psych(df_trial, 'choice', axs0[2], legend = 'Fit', color = 'r', ls = '--')
axs0[2].legend()

# plot history_dependent psychometric fits
if len(np.unique(df_trial['history_bias'])) > 1:
    plot_psych(df_trial.query('history_bias == 2'), 'choice_target_end', axs0[3], legend = 'Post right', color = 'hotpink')
    plot_psych(df_trial.query('history_bias == -2'), 'choice_target_end', axs0[3], legend = 'Post left', color = 'deepskyblue')
    plot_psych(df_trial.query('history_bias == 2'), 'choice', axs0[3], color = 'hotpink', ls = '--')
    plot_psych(df_trial.query('history_bias == -2'), 'choice', axs0[3], color = 'deepskyblue', ls = '--')
    axs0[3].legend()
    
    

axs1 = subfigs[1].subplots(3,4)

# plot a correct trial (same as target)
trialnum = np.random.choice(np.where((df_trial.choice == df_trial.choice_target_end) & (df_trial.is_lapse == 0))[0])
plot_trial(axs1[0,:], data_dict, df_trial, trialnum, FOF_ADS.dt, legend = True)

# plot an incorrect trial (opposite of target)
trialnum = np.random.choice(np.where((df_trial.choice != df_trial.choice_target_end) & (df_trial.is_lapse == 0))[0])
plot_trial(axs1[1,:], data_dict, df_trial, trialnum, FOF_ADS.dt)

if len(np.unique(df_trial['is_lapse'])) > 1:
    # plot a lapse trial
    trialnum = np.random.choice(np.where(df_trial.is_lapse == 1)[0])
    plot_trial(axs1[2,:], data_dict, df_trial, trialnum, FOF_ADS.dt)
else:
    # plot a correct trial (same as target)
    trialnum = np.random.choice(np.where((df_trial.choice == df_trial.choice_target_end) & (df_trial.is_lapse == 0))[0])
    plot_trial(axs1[2,:], data_dict, df_trial, trialnum, FOF_ADS.dt, legend = True)

    
sns.despine()

savethisfig(dirname, 'model_training')





# VISUALIZE INACTIVATION EFFECTS
new_params = {'N_batch': 2000,
              'p_probe': 1.,
              'frac_opto': 0.,
              'probe_duration': 1000}

FOF_ADS.destruct()

FOF_ADS, pc_data, model_sim = reinitialize_network(dirname, new_params)

# generate trials and run RNN
x,y,m,params = pc_data.get_trial_batch()
output, activity = FOF_ADS.test(x)
df_trial_cntrl, activity_cntrl, data_dict = format_data(x,y,m, params, output, activity)


perturb_group = ['left_FOF', 'right_FOF','bi_FOF', 'left_ADS', 'right_ADS', 'left_proj', 'right_proj', 'left_ADSproj', 'right_ADSproj']
perturb_type = ['first_half', 'second_half']
gain_ranges = [0.1]

n_grps = len(perturb_group)
n_conds = len(perturb_type) * len(gain_ranges)
fig, axs = plt.subplots(n_grps, n_conds, figsize = (2.5*n_conds, 2.5*n_grps))

for p, grp in enumerate(perturb_group):
    for t, ptype in enumerate(perturb_type):
        print(grp, ptype)
        for g, gain in enumerate(gain_ranges):
            t_connectivity = make_perturbation_inputs_mul(FOF_ADS.N_steps, 
                                          perturb_type = ptype, 
                                          perturb_group = grp, 
                                          gain = gain,
                                          dt = FOF_ADS.dt)
            output, activity = model_sim.run_trials(x,t_connectivity = t_connectivity)
            df_trial_inact,_,_ = format_data(x,y,m,params,output, activity)
            plot_psych(df_trial_cntrl, 'choice',axs[p, len(gain_ranges)*t + g], color = 'k', ls = '--')
            plot_psych(df_trial_inact, 'choice', axs[p, len(gain_ranges)*t + g], color = 'r', ls = '--')
            axs[p, len(gain_ranges)*t + g].set_title(grp + ' ' + ptype + ' | ' + str(gain), fontsize = 8)
            
         
plt.tight_layout()
sns.despine()

savethisfig(dirname, 'model_inactivation')



