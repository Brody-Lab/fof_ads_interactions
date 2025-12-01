import re, sys
basedir = '/home/dikshag/pbups_ephys_analysis/pbups_phys/multiregion_RNN/'
sys.path.append(basedir + '/base/')
sys.path.append(basedir)
from multiregion_RNN_utils import *

base_path = get_base_path()
figure_dir = get_figure_dir()

fitname = sys.argv[1]

# fetch filenames
files = sorted([fn for fn in os.listdir(base_path) if re.findall(fitname, fn) and not fn.endswith(".npy")])
print(files)

# which projections/areas to perturb
perturb_group = ['bi_FOF', 
                 'left_proj', 
                 'right_proj',
                 'left_FOF', 
                 'right_FOF',
                 'left_ADS',
                 'right_ADS']

# what kind of perturbations
perturb_type = ['first_half', 'second_half'] 

# gains at which to inactivate
gain_ranges = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# set all trials to be probe trials for these tests
new_params = {'N_batch': 1000, 
              'p_probe': 1., 
              'probe_duration': 1000}

# initializing data structure for storing 
summary = {pgrp:{ptype: {g: {b :[] for b in ['bias','accuracy']} for g in gain_ranges} for ptype in perturb_type} for pgrp in perturb_group}


# loop through
for file in files:
    
    # reinitialize network
    FOF_ADS, pc_data, model_sim = reinitialize_network(file, new_params)
      
    # generate trials and run RNN
    x_inp, y_inp, m_inp, params = pc_data.get_trial_batch()
    output, activity = FOF_ADS.test(x_inp)
    df_trial, _, _ = format_data(x_inp, y_inp, m_inp, params, output, activity)
    fracR = compute_fracR(df_trial)
    accuracy = np.mean(df_trial['choice'] == df_trial['correct'])

    for pgrp in perturb_group:
        for ptype in perturb_type:
            print(pgrp, ptype)
            for gain in gain_ranges:
                
                # run network with Wrec perturbed
                t_connectivity = make_perturbation_inputs_mul(FOF_ADS.N_steps, 
                                          perturb_type = ptype, 
                                          perturb_group = pgrp, 
                                          gain = gain,
                                          dt = FOF_ADS.dt)
                output, activity = model_sim.run_trials(x_inp, t_connectivity = t_connectivity)
                df_trial_in, _, _ = format_data(x_inp, y_inp, m_inp, params, output, activity)
                fracR_in = compute_fracR(df_trial_in)
                accuracy_in = np.mean(df_trial_in['choice'] == df_trial_in['correct'])
                
                # plt.figure()
                # plot_psych(df_trial, 'choice', plt.gca(), color = 'k', ls = '--')
                # plot_psych(df_trial_in, 'choice', plt.gca(), color = 'r', ls = '--')
                # plt.title(pgrp + ptype)
                # plt.show()
                
                # compute change in ipsi bias and accuracy
                if 'right' in pgrp:
                    summary[pgrp][ptype][gain]['bias'].append(-1*np.mean(fracR - fracR_in))
                else:
                    summary[pgrp][ptype][gain]['bias'].append(np.mean(fracR - fracR_in))
                summary[pgrp][ptype][gain]['accuracy'].append(accuracy - accuracy_in)
                
    FOF_ADS.destruct()
                                                            
                
np.save(base_path + fitname + '_inactivation_train_summary_' + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".npy", summary)

