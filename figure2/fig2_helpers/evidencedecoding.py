import sys
sys.path.insert(1, '../../../figure_code/')
from my_imports import *

from tqdm import tqdm
import random
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from helpers.phys_helpers import get_sortCells_for_rat, equalize_neurons_across_regions, datetime
from helpers.physdata_preprocessing import load_phys_data_from_Cell
from helpers.rasters_and_psths import get_neural_activity


def get_evidence(ntpts_per_trial, df_trial, p, ev_type="delta"):
    """_summary_

    Args:
        ntpts_per_trial (_type_): _description_
        df_trial (_type_): _description_
        p (_type_): _description_
        ev_type (str, optional): _description_. Defaults to "delta".

    Returns:
        _type_: _description_
    """
    
    print("aligning to clicks on ignoring definition in p")
    max_tpts = np.max(ntpts_per_trial)
    evidence = np.nan * np.zeros((len(df_trial), max_tpts))
    for tr, ntpts in zip(range(len(df_trial)), ntpts_per_trial):
        edges = np.arange(p['start_time'][0], (ntpts+1)*p['binsize'], p['binsize'])*0.001
        counts_L, _ = np.histogram(df_trial['leftBups'][tr] - df_trial['clicks_on'][tr], edges)
        counts_R, _ = np.histogram(df_trial['rightBups'][tr] - df_trial['clicks_on'][tr], edges)
        if ev_type == "delta":
            evidence[tr, :ntpts] = np.cumsum(counts_R - counts_L)
        elif ev_type == "right":
            evidence[tr, :ntpts] = np.cumsum(counts_R)
        elif ev_type == "left":
            evidence[tr, :ntpts] = np.cumsum(counts_L)
    return evidence


def run_evidence_decoding(X, evidence, ntpts_per_trial, p):    
    """This function runs the evidence decoding analysis

    Args:
        X (array): _description_
        evidence (array): _description_
        ntpts_per_trial (int): _description_
        p (dict): _description_

    Returns:
        _type_: _description_
    """
    evidence = np.ravel(evidence)
    Y_evidence = evidence[~np.isnan(evidence)]
    cv = KFold(n_splits=p['nfolds'], shuffle = True)
    if len(p['Cs']) == 1:
        mdl = LinearRegression()
        print("here")
    else:
        mdl = LassoCV(
            eps = p['Cs'][-1]/p['Cs'][0],
            cv = cv,
            alphas = p['Cs'],
            max_iter = int(1e8),
            tol = 1e-6)
        
    # repeat the decoding for n_repeats to get CI over accuracies
    t0_index = np.hstack(([0], np.cumsum(ntpts_per_trial)[:-1]))
    max_tpts = max(ntpts_per_trial)
    accuracy_fold = np.nan * np.zeros((p['n_repeats'], max_tpts))
    print(np.shape(Y_evidence), np.shape(X))
    for it in range(p['n_repeats']):
        y_pred = cross_val_predict(mdl, X.T, Y_evidence, cv = cv)
        # find model accuracy at each time point
        for t in range(max_tpts):
            tr_idx = np.squeeze(np.where(ntpts_per_trial > t))
            if tr_idx.size > 50:
                accuracy_fold[it, t] = np.corrcoef(
                    y_pred[t0_index[tr_idx] + t], 
                    Y_evidence[t0_index[tr_idx] + t])[0,1]
                
    accuracy_sem = np.nan*np.zeros((max_tpts,2))
    accuracy_std = np.nan*np.zeros(max_tpts)
    for t in range(max_tpts):
        accuracy_sem[t,:] = [np.percentile(accuracy_fold[:,t], 2.5),
                            np.percentile(accuracy_fold[:,t], 97.5)]
        accuracy_std[t] = np.std(accuracy_fold[:,t])
        
    mdl.fit(X.T, Y_evidence)
    results = dict()
    results['accuracy'] = np.nanmean(accuracy_fold, axis = 0)
    results['accuracy_sem'] = accuracy_sem
    results['accuracy_fold'] = accuracy_fold
    results['pred_evidence'] =  y_pred
    results['true_delta_clicks'] = Y_evidence
    results['tpts'] = range(max_tpts)
    results['mdl_coefs'] = mdl.coef_
    results['accuracy_std'] = accuracy_std
    return results


def plot_evidence_decoding(res, ax, p):
    """_summary_

    Args:
        res (_type_): _description_
        ax (_type_): _description_
        p (_type_): _description_
    """
    cols = p['cols']
    for reg in p['regions']:
        t = [1e-3*(p['binsize'] * it + p['binsize']/2 + p['Tshift']) for it in res[reg]['tpts']]
        ax.plot(t, res[reg]['accuracy'], label = reg, c= cols[reg])
        ax.fill_between(t, 
                        res[reg]['accuracy_sem'][:,0], 
                        res[reg]['accuracy_sem'][:,1], 
                        alpha = 0.5, 
                        color = cols[reg])
        ax.legend(frameon = False, handlelength = 1)
    ax.axhline(0.0, c = 'k', lw = 0.5, ls = ':')
    ax.set_ylim([-0.2, 0.85])
    ax.set_xlabel('Time from stimulus on [s]')
    ax.set_ylabel('corr(evidence, predicted)')
    ax.grid()
    
    
if __name__ == "__main__":
    
    p = dict()
    p['ratnames'] = SPEC.RATS
    p['regions'] = SPEC.REGIONS
    p['cols'] = SPEC.COLS
    p['fr_thresh'] = 1.0  # firing rate threshold for including neurons
    p['stim_thresh'] = 0.0  # stimulus duration threshold for including trials
    p['align_to'] = ['clicks_on', 'clicks_on']
    p['align_name'] = ['clickson_masked', 'clickson_unmasked']
    p['pre_mask'] = [None, None]
    p['post_mask'] = ['clicks_off_delayed', None]
    p['start_time'] = [0,0]
    p['end_time'] = [1100, 1100] # in ms   
    p['trial_type'] = ['all', 'right_going', 'left_going']
    p['Tshift'] = 100 # delay for stimulus (in ms)
    p['binsize'] = 50  # in ms
    p['filter_type'] = 'gaussian'
    p['filter_w'] = 75  # in ms
    p['Cs'] = np.logspace(-8,8,400)  # cross-validation parameter
    p['nfolds'] = 10    # number of folds for cross-validation
    p['n_repeats'] = 10 # number of repeats for cross-validation
    
    SAVEDIR = SPEC.RESULTDIR + 'evidence_decoding/'
    fname = SAVEDIR + 'params' + datetime()[5:] + ".npy"
    np.save(fname, p)
    
    rat = p['ratnames'][int(sys.argv[1])]
    
    print("\n\n\n\n===== RAT: {} =====".format(rat))
    files, this_datadir = get_sortCells_for_rat(rat, SPEC.DATADIR)
    
    file = files[int(sys.argv[2])]
    

    print('\n\nProcessing file: {}'.format(file))
    fname = SAVEDIR + file[:21] + datetime()[5:] + '.npy'
    df_trial, df_cell, _ = load_phys_data_from_Cell(this_datadir + os.sep + file)
    df_trial = df_trial[df_trial['stim_dur_s_actual'] >= p['stim_thresh']].reset_index(drop = True)
    df_trial['clicks_on_delayed'] = df_trial['clicks_on'] + 1e-3*p['Tshift']
    df_trial['clicks_off_delayed'] = df_trial['clicks_off'] + 1e-3*p['Tshift']

    # equalize left and right choice trials
    random.seed(10)
    n = df_trial['pokedR'].value_counts().min()
    idx_R = random.sample(list(np.where(df_trial['pokedR'] == 1)[0]), n)[:n]
    idx_L = random.sample(list(np.where(df_trial['pokedR'] == 0)[0]), n)[:n]
    df_trial = df_trial.loc[np.sort(idx_R + idx_L)].reset_index(drop = True)
    ntrials = len(df_trial)
    print("trial_count: {}".format(ntrials))

    # equalize neurons across regions
    df_cell = df_cell[df_cell['stim_fr'] >= p['fr_thresh']].reset_index(drop = True)
    df_cell = equalize_neurons_across_regions(df_cell, p['regions'])

    summary = dict()
    for align_id, (align, align_name) in enumerate(zip(p['align_to'], p['align_name'])):
        print("\n\nProcessing alignment: {}".format(align_name))
        align_to = p['align_to'][align_id]
        summary[align_name] = dict()
        for tr_type in p['trial_type']:
            summary[align_name][tr_type] = dict()

        for r, reg in enumerate(p['regions']):
            print("\n\tProcessing region: {}".format(reg))

            for e, tr_type in enumerate(p['trial_type']):
                if tr_type == 'all':
                    this_df_trial = df_trial
                elif tr_type == 'right_going':
                    this_df_trial = df_trial[df_trial['pokedR'] == 1].reset_index()
                elif tr_type == 'left_going':
                    this_df_trial = df_trial[df_trial['pokedR'] == 0].reset_index()
                else:
                    print("ERROR: trial type not recognized")

                X, ntpts_per_trial = get_neural_activity(df_cell, this_df_trial, reg, p, align_id)
                evidence = get_evidence(ntpts_per_trial, this_df_trial, p)
                print("\n running trial type: {}".format(tr_type))

                summary[align_name][tr_type][reg] = run_evidence_decoding(X, evidence, ntpts_per_trial, p)


    summary['prm'] = dict()
    summary['prm']['filename'] = file
    summary['prm']['ntrials'] = len(df_trial)
    summary['prm']['n_neurons'] = [df_cell['region'].value_counts()['ADS'], df_cell['region'].value_counts()['FOF']]
    np.save(fname, summary)
    
