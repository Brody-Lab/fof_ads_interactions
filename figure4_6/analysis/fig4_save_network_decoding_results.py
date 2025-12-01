import sys, re, warnings
basedir = '/home/dikshag/pbups_ephys_analysis/pbups_phys/'
sys.path.append(basedir)
sys.path.append(basedir + 'multiregion_RNN/')
from multiregion_RNN_utils import *
from plotting_general import *
from dev_nov21_trialhistory_decoding_helpers import stable_choice_decode
from dev_nov21_evidencedecoding_helpers import stable_evidence_decode




def format_activity_for_decoding(df_trial, activity):

    stim_on = (df_trial['onset_time']/FOF_ADS.dt).astype(int) 
    stim_off = ((df_trial['onset_time']+df_trial['stim_duration'])/FOF_ADS.dt).astype(int) 
    ntrials = len(df_trial)
    num_cells = np.shape(activity)[0]

    X = [] 

    for i in range(num_cells):
        fr_align = [activity[i][tr, stim_on[tr]:stim_off[tr]] for tr in range(ntrials)]
        fr_align = np.array(list(itertools.zip_longest(*fr_align,fillvalue=np.nan))).T

        # remove mean PSTH
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fr_align = fr_align - np.nanmean(fr_align, axis = 0)

        X.append(np.ravel(fr_align))

    ntpts_per_trial = [sum(~np.isnan(fr_align[tr])) for tr in range(ntrials)]
    
    X = np.array(X)
    X = X[:, ~np.all(np.isnan(X), axis = 0)]

    # z-score each feature
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X.T).T

    return X, np.array(ntpts_per_trial)



def get_evidence(df_trial, ntpts, ev_type = 'delta'):

    max_tpts = np.max(ntpts)
    evidence = np.nan * np.zeros((len(df_trial), max_tpts))

    for tr, tpt in zip(range(len(df_trial)), ntpts):
        edges = np.arange(0, (tpt+1)*10, 10) # 10ms is the binsize
        counts_L, _ = np.histogram(df_trial['left_clicks'][tr] - df_trial['onset_time'][tr], edges)
        counts_R, _ = np.histogram(df_trial['right_clicks'][tr] - df_trial['onset_time'][tr], edges)

        if ev_type == "delta":
            evidence[tr, :tpt] = np.cumsum(counts_R - counts_L)
        elif ev_type == "right":
            evidence[tr, :tpt] = np.cumsum(counts_R)
        elif ev_type == "left":
            evidence[tr, :tpt] = np.cumsum(counts_L)
                
    return evidence



if __name__ == "__main__":
    
    # define which fits to decode from
    fitname = sys.argv[1]
    base_path = get_base_path()

    # fetch files
    files = sorted([fn for fn in os.listdir(base_path) if re.findall(fitname, fn) and not fn.endswith(".npy")])

    # Cross validating settings for decoding
    p = dict()
    p['choice'] = {'nfolds': 3,
                   'Cs': np.logspace(-7,3,200),
                   'n_repeats': 1}
    p['stim'] = {'nfolds': 3,
                 'Cs': np.logspace(-8,8,400),
                 'n_repeats': 1}


    # new parameters for reinitializing RNN (increasing batch size)
    new_params = {'N_batch': 300, 
                  'p_probe': 0., 
                  # 'frac_opto': 0.,
                  'probe_duration': 0}



    # initialize data structure for saving
    summary = dict()
    for s in ['stim', 'choice']:
        summary[s] = dict()
        for reg in ['FOF', 'ADS']:
            summary[s][reg] = dict()


    for file in files:

        # reinitialize network
        FOF_ADS, pc_data, _ = reinitialize_network(file, new_params)

        # generate trials and run RNN
        x,y,m,params = pc_data.get_trial_batch()
        output, activity = FOF_ADS.test(x)
        df_trial, activity, data_dict = format_data(x,y,m,params, output, activity)

        X_ch, ntpts = format_activity_for_decoding(df_trial, activity)

        # decode choice 
        print('\t\t' + '- decoding choice from "FOF"')
        summary['choice']['FOF'][file] = stable_choice_decode(
            X_ch[:100,:], 
            1*df_trial.choice, 
            ntpts, 
            p['choice'])

        print('\t\t' + '- decoding choice from "ADS"')
        summary['choice']['ADS'][file] = stable_choice_decode(
            X_ch[200:,:], 
            1*df_trial.choice, 
            ntpts, 
            p['choice'])


        # decode Δclicks
        evidence = get_evidence(df_trial, ntpts)
        print('\t\t' + '- decoding Δclicks from "FOF"')
        summary['stim']['FOF'][file] = stable_evidence_decode(
            X_ch[:100,:], 
            evidence, 
            ntpts, 
            p['stim'])

        print('\t\t' + '- decoding Δclicks from "ADS"')
        summary['stim']['ADS'][file] = stable_evidence_decode(
            X_ch[200:,:], 
            evidence, 
            ntpts, 
            p['stim'])

        FOF_ADS.destruct()




    np.save(base_path + fitname + '_choice_stim_decoding_' + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".npy", summary)

