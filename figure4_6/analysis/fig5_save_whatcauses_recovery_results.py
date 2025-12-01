import re, sys
basedir = '/home/dikshag/pbups_ephys_analysis/pbups_phys/multiregion_RNN/'
sys.path.append(basedir + '/base/')
sys.path.append(basedir)
from multiregion_RNN_utils import *
from itertools import chain, combinations

base_path = get_base_path()
figure_dir = get_figure_dir()

fitname = sys.argv[1]

# fetch filenames
files = sorted([fn for fn in os.listdir(base_path) if re.findall(fitname, fn) and not fn.endswith(".npy")])
print(files)


# which projections/areas to perturb
from_neurons = {'leftFOF' : range(40),
               'rightFOF': range(40,80)}

to_neurons = {'leftADS': range(200, 250),
              'rightADS': range(250, 300),
              'leftREC': range(40),
              'rightREC': range(40,80),
              'rightFOF': [*range(40,80), *range(190,200)],
              'leftFOF': [*range(40), *range(180,190)]}

exclude_comb = [['leftFOF', 'rightFOF'],
                ['leftFOF', 'leftREC'],
                ['rightFOF', 'rightREC'], 
                ['leftREC', 'rightREC']]

all_targets = [
    '_'.join(map(str, comb))
    for r in range(1, len(to_neurons) + 1)
    for comb in combinations(to_neurons, r)
    if not any(set(exclude).issubset(comb) for exclude in exclude_comb)
]

# what kind of perturbations
perturb_type = ['first_half', 'second_half'] 

# gains at which to inactivate
gains = [0.1]

# set all trials to be probe trials for these tests
new_params = {'N_batch': 1000, 
              'p_probe': 1., 
              'probe_duration': 1000}

# initializing data structure for storing 
summary = dict()


def make_perturbation_inputs_mul(N_steps, 
                                 perturb_type, 
                                 source_group, 
                                 target_group, 
                                 dt = 10, 
                                 gain = 0.1):
    
    t_connectivity = np.ones((300,300))
    t_connectivity_perturb = np.ones((300,300))
    
    target_idx = list(chain(*[to_neurons[k] for k in target_group.split('_')]))
    which_neurons = np.ix_(target_idx, from_neurons[source_group])    
    
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
    
    t_connectivity_perturb[which_neurons] = gain    
    t_connectivity= np.repeat(t_connectivity[np.newaxis, :, :], N_steps, axis = 0)
    t_connectivity[t0:t1] = t_connectivity_perturb
    
    return t_connectivity


def rearrange_string(input_string):
    terms = input_string.split('_')
    
    ipsi_terms = sorted([term for term in terms if 'ipsi' in term])
    contra_terms = sorted([term for term in terms if 'contra' in term])

    result_string = '_'.join(ipsi_terms + contra_terms)

    return result_string
    
    
    
# loop through
for file in files:
    
    print(file)
    
    # reinitialize network
    FOF_ADS, pc_data, model_sim = reinitialize_network(file, new_params)
      
    # generate trials and run RNN
    x_inp, y_inp, m_inp, params = pc_data.get_trial_batch()
    output, activity = FOF_ADS.test(x_inp)
    df_trial, _, _ = format_data(x_inp, y_inp, m_inp, params, output, activity)
    fracR = compute_fracR(df_trial)
    accuracy = np.mean(df_trial['choice'] == df_trial['correct'])


    for source in from_neurons:
        for target in all_targets:
            
            if (source == 'leftFOF'): 
                side = "left"
                contraside = "right"
                if ('rightREC' in target) or ('leftFOF' in target):
                    continue
            
            if (source == 'rightFOF'):
                side = "right"
                contraside = "left"
                if('leftREC' in target) or ('rightFOF' in target):
                    continue
                    
            if side in target:
                target_name = target.replace(side, "ipsi")
            else:
                target_name = target
                
            if contraside in target:
                target_name = target_name.replace(contraside, "contra")
                
            target_name = rearrange_string(target_name)
                
            # Check if the target_name key exists in the top-level dictionary
            if target_name not in summary:
                summary[target_name] = {}
            
            
            for pt, ptype in enumerate(perturb_type):
                
                # Check if the ptype key exists in the second-level dictionary
                if ptype not in summary[target_name]:
                    summary[target_name][ptype] = {}
                    
                print(source, target, ptype, target_name)

                
                for gain in gains:
                    
                    # Check if the gain key exists in the third-level dictionary
                    if gain not in summary[target_name][ptype]:
                        summary[target_name][ptype][gain] = {'bias': []}

                    t_connectivity = make_perturbation_inputs_mul(FOF_ADS.N_steps, 
                                              perturb_type = ptype, 
                                              source_group = source, 
                                              target_group = target,
                                              gain = gain,
                                              dt = FOF_ADS.dt)
                    
                    output, activity = model_sim.run_trials(x_inp, 
                                                            t_connectivity = t_connectivity)
                    df_trial_in, _, _ = format_data(x_inp, 
                                                    y_inp, 
                                                    m_inp, 
                                                    params, 
                                                    output, 
                                                    activity)
                    fracR_in = compute_fracR(df_trial_in)
                    accuracy_in = np.mean(df_trial_in['choice'] == df_trial_in['correct'])
                    
                    if side == "right":
                        bias = -1*np.mean(fracR - fracR_in)
                    else:
                        bias = np.mean(fracR - fracR_in) 

                    summary[target_name][ptype][gain]['bias'].append(bias)
                       
            
    FOF_ADS.destruct()
    
np.save(base_path + fitname + '_whatcauses_recovery_summary_' + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".npy", summary)



                 

