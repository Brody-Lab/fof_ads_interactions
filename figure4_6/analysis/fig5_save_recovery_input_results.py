import re, sys
basedir = '/home/dikshag/pbups_ephys_analysis/pbups_phys/multiregion_RNN/'
sys.path.append(basedir)
from multiregion_RNN_utils import *


"""
What is necessary for recovery in the first half

To test this we will turn off these possible inputs in 2nd half post inactivation
1. inputs from the relay
2. auditory stimulus
3. inputs from the other hemisphere

"""

def compute_ipsi_bias_dueto_recovery(file, ipsi_bias, plotting = False):
    
    
    new_params = {'N_batch': 1000,
                  'p_probe': 1.0,
                  'probe_duration': 1000}

    FOF_ADS, pc_data, model_sim = reinitialize_network(file, new_params)

    gain = 0.1
    stim_start = int(500/FOF_ADS.dt)
    stim_mid = int(1000/FOF_ADS.dt)
    stim_end = int(1500/FOF_ADS.dt)

    # generate trials and run RNN
    x_inp, y_inp, m_inp, params = pc_data.get_trial_batch()
    output, activity = model_sim.run_trials(x_inp)
    df_trial, _, _ = format_data(x_inp, y_inp, m_inp, params, output, activity)
    fracR = compute_fracR(df_trial)



    def get_ipsi_bias(x_inp, y_inp, m_inp, params, fracR, df_trial, plotting, t_connectivity):

        output, activity = model_sim.run_trials(x_inp, t_connectivity = t_connectivity)
        this_df_trial, _, _ = format_data(x_inp, y_inp, m_inp, params, output, activity)
        this_df_trial["Δclicks"] = df_trial["Δclicks"]    
        this_fracR = compute_fracR(this_df_trial)

        if_plotting(plotting, df_trial, this_df_trial)

        if side == 'right':
            return -1 * np.mean(fracR - this_fracR)
        else:
            return np.mean(fracR - this_fracR)


    def if_plotting(plotting, df_trial, this_df_trial):
         if plotting:
            plt.figure()
            plot_psych(df_trial, 'choice', plt.gca(), color = 'k', ls = '--')
            plot_psych(this_df_trial, 'choice', plt.gca(), color = 'r', ls = '--')
            plt.show()  


    def get_t_change(side, input_type, gain = 0.1):

        t_change = np.ones((300,300))
        if input_type == "relay":
            if side == "left":
                t_change[np.ix_([*range(40),*range(180,190)], range(80,180))] = gain
            else:
                t_change[np.ix_([*range(40,80),*range(190,200)], range(80, 180))] = gain  

        elif input_type == "other":
            if side == "left":
                t_change[np.ix_([*range(40), *range(180,190)], range(40,80))] = gain
            else:
                t_change[np.ix_([*range(40,80), *range(190,200)], range(40))] = gain

        return t_change



    recovery_keys = ['uni_FOF', 
                     'other_FOF', 
                     'relay_FOF', 
                     'stim_FOF', 
                     'other_n_relay_FOF', 
                     'stim_n_other_FOF', 
                     'stim_n_relay_FOF']

    for key in recovery_keys:
        ipsi_bias.setdefault(key, [])



    # first get ipsilateral bias from first half FOF inactivation (average across left and right)
    print("Computing for baseline recovery")
    this_ipsi_bias = []
    for side in ['left', 'right']:
        t_connectivity = make_perturbation_inputs_mul(
            FOF_ADS.N_steps,
            perturb_type = 'first_half',
            perturb_group = side + '_FOF',
            gain = gain,
            dt = FOF_ADS.dt)  
        this_ipsi_bias.append(get_ipsi_bias(x_inp, 
                                            y_inp, 
                                            m_inp, 
                                            params,
                                            fracR,
                                            df_trial,
                                            plotting,
                                            t_connectivity))


    ipsi_bias['uni_FOF'].append(sum(this_ipsi_bias) / len(this_ipsi_bias))



    print("Computing for other FOF's inputs")
    # ipsilateral bias from when inputs from other FOF are turned off in second half
    this_ipsi_bias = []
    for side in ['left', 'right']:
        t_connectivity =  make_perturbation_inputs_mul(
            FOF_ADS.N_steps,
            perturb_type = 'first_half',
            perturb_group = side + '_FOF',
            gain = gain,
            dt = FOF_ADS.dt)       
        t_connectivity[stim_mid:stim_end] = get_t_change(side, "other")
        this_ipsi_bias.append(get_ipsi_bias(x_inp, 
                                            y_inp, 
                                            m_inp, 
                                            params,
                                            fracR,
                                            df_trial,
                                            plotting,
                                            t_connectivity))


    ipsi_bias['other_FOF'].append(sum(this_ipsi_bias) / len(this_ipsi_bias))


    print("Computing for relay inputs")
    # ipsilateral bias from when relay inputs are turned off
    this_ipsi_bias = []
    for side in ['left', 'right']:
        t_connectivity =  make_perturbation_inputs_mul(
            FOF_ADS.N_steps,
            perturb_type = 'first_half',
            perturb_group = side + '_FOF',
            gain = gain,
            dt = FOF_ADS.dt)
        t_connectivity[stim_mid:stim_end] = get_t_change(side, "relay")
        this_ipsi_bias.append(get_ipsi_bias(x_inp, 
                                            y_inp, 
                                            m_inp, 
                                            params,
                                            fracR,
                                            df_trial,
                                            plotting,
                                            t_connectivity))


    ipsi_bias['relay_FOF'].append(sum(this_ipsi_bias) / len(this_ipsi_bias))



    print("Computing for no relay and other inputs")
    # ipsilateral bias from when relay inputs are turned off
    this_ipsi_bias = []
    for side in ['left', 'right']:
        t_connectivity =  make_perturbation_inputs_mul(
            FOF_ADS.N_steps,
            perturb_type = 'first_half',
            perturb_group = side + '_FOF',
            gain = gain,
            dt = FOF_ADS.dt)
        t_change = np.ones((300,300))
        if side == "left":
            t_change[np.ix_([*range(40),*range(180,190)], range(80,180))] = gain
            t_change[np.ix_([*range(40), *range(180, 190)], range(40,80))] = gain
        else:
            t_change[np.ix_([*range(40,80),*range(190,200)], range(80, 180))] = gain  
            t_change[np.ix_([*range(40,80), *range(190,200)], range(40))] = gain
        t_connectivity[stim_mid:stim_end] = t_change
        this_ipsi_bias.append(get_ipsi_bias(x_inp, 
                                            y_inp, 
                                            m_inp, 
                                            params,
                                            fracR,
                                            df_trial, plotting,
                                            t_connectivity))


    ipsi_bias['other_n_relay_FOF'].append(sum(this_ipsi_bias) / len(this_ipsi_bias))




    print("Computing for no stimulus")
    # ipsilateral bias when inputs in 2nd half are turned off
    this_ipsi_bias = []

    for i in range(FOF_ADS.N_batch):
        x_inp[i, stim_mid:stim_end, :] = 0

    output, activity = model_sim.run_trials(x_inp)
    df_trial, _, _ = format_data(x_inp, y_inp, m_inp, params, output, activity)
    for i in range(len(df_trial)):
        df_trial["Δclicks"][i] = sum((df_trial['right_clicks'][i] < 1000) == True) - sum((df_trial['left_clicks'][i] < 1000) == True)
    fracR = compute_fracR(df_trial)



    for side in ['left', 'right']:

        t_connectivity =  make_perturbation_inputs_mul(
            FOF_ADS.N_steps,
            perturb_type = 'first_half',
            perturb_group = side + '_FOF',
            gain = gain,
            dt = FOF_ADS.dt)

        # compute with uniFOF inactivation
        this_ipsi_bias.append(get_ipsi_bias(x_inp, 
                                            y_inp, 
                                            m_inp, 
                                            params,
                                            fracR,
                                            df_trial, 
                                            plotting,
                                            t_connectivity))


    ipsi_bias['stim_FOF'].append(sum(this_ipsi_bias) / len(this_ipsi_bias))



    print("Computing for no stimulus and no relay")
    # ipsilateral bias when inputs in 2nd half are turned off
    this_ipsi_bias = []
    for side in ['left', 'right']:
        t_connectivity =  make_perturbation_inputs_mul(
            FOF_ADS.N_steps,
            perturb_type = 'first_half',
            perturb_group = side + '_FOF',
            gain = gain,
            dt = FOF_ADS.dt)
        t_connectivity[stim_mid:stim_end] = get_t_change(side, "relay")
        # compute with uniFOF inactivation
        this_ipsi_bias.append(get_ipsi_bias(x_inp, 
                                            y_inp, 
                                            m_inp, 
                                            params,
                                            fracR,
                                            df_trial,
                                            plotting,
                                            t_connectivity))


    ipsi_bias['stim_n_relay_FOF'].append(sum(this_ipsi_bias) / len(this_ipsi_bias))



    print("Computing for no stimulus and no other FOF inputs")
    # ipsilateral bias when inputs in 2nd half are turned off
    this_ipsi_bias = []
    for side in ['left', 'right']:
        t_connectivity =  make_perturbation_inputs_mul(
            FOF_ADS.N_steps,
            perturb_type = 'first_half',
            perturb_group = side + '_FOF',
            gain = gain,
            dt = FOF_ADS.dt)
        t_connectivity[stim_mid:stim_end] = get_t_change(side, "other")
        # compute with uniFOF inactivation
        this_ipsi_bias.append(get_ipsi_bias(x_inp, 
                                            y_inp, 
                                            m_inp, 
                                            params,
                                            fracR,
                                            df_trial, 
                                            plotting,
                                            t_connectivity))


    ipsi_bias['stim_n_other_FOF'].append(sum(this_ipsi_bias) / len(this_ipsi_bias))



    try:
        FOF_ADS.destruct()
    except:
        print("nothing to destruct")

    return ipsi_bias






if __name__ == "__main__":
    
    base_path = get_base_path()
    
    fitfile = sys.argv[1]
    files = sorted([fn for fn in os.listdir(base_path) if re.findall(fitfile, fn) and not fn.endswith(".npy")])
    
    ipsi_bias = dict()
    for file in files:
        print("\n" + file)
        try:
            ipsi_bias = compute_ipsi_bias_dueto_recovery(file, ipsi_bias, plotting = False)
        except:
            FOF_ADS.destruct()
            ipsi_bias = compute_ipsi_bias_dueto_recovery(file, ipsi_bias, plotting = False)

    np.save(base_path + fitfile + '_uniFOF_recovery_whichinputs_' + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".npy", ipsi_bias)