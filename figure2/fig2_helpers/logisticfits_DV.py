import sys
sys.path.insert(1, '../../../figure_code/')
from my_imports import *

from tqdm import tqdm
import random
from helpers.phys_helpers import get_sortCells_for_rat, equalize_neurons_across_regions, datetime
from helpers.rasters_and_psths import get_neural_activity
from helpers.physdata_preprocessing import load_phys_data_from_Cell
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from logisticdecoding import equalize_trials



def run_logistic_decoding(X, target, ntpts_per_trial, p):
    
    Y = np.repeat(target, ntpts_per_trial)
    cv_reg = KFold(n_splits=p['nfolds'], shuffle = True)
    mdl = LogisticRegressionCV(
        cv = cv_reg,
        Cs = p['Cs'],
        max_iter = 1e8,
        tol = 1e-4
    )
    cv_repeat = KFold(n_splits=p['nfolds'], shuffle = True)
    
    # repeat k-fold crossvalidation to get CIs
    t0_index = np.hstack(([0], np.cumsum(ntpts_per_trial)[:-1]))
    max_tpts = max(ntpts_per_trial)
    accuracy_fold = np.nan * np.zeros((p['n_repeats'], max_tpts))
    
    cv_dict = dict()
    idx_trial = np.concatenate(([0], np.cumsum(ntpts_per_trial)))

    
    for it in range(p['n_repeats']):
        
        cv_dict[it] = dict()
        cv_dict[it]['mdl_coefs'] = np.nan * np.zeros((p['nfolds'], np.shape(X)[0]))
        cv_dict[it]['mdl_intercept'] = np.nan * np.zeros((p['nfolds']))
        cv_dict[it]['test_trials'] = []
        y_pred = np.nan * np.zeros(np.shape(Y))
        
        for fold, (train_idx, test_idx) in enumerate(cv_repeat.split(ntpts_per_trial)):
            
            slices = [slice(start, end ) for start, end in zip(idx_trial[train_idx], idx_trial[train_idx+1])]
            train_index = np.concatenate([np.arange(s.start, s.stop) for s in slices])
        
            slices = [slice(start, end ) for start, end in zip(idx_trial[test_idx], idx_trial[test_idx+1])]
            test_index = np.concatenate([np.arange(s.start, s.stop) for s in slices])
        
            mdl.fit(X.T[train_index], Y[train_index])
            y_pred[test_index] = mdl.predict(X.T[test_index])
            cv_dict[it]['mdl_coefs'][fold, :] = mdl.coef_
            cv_dict[it]['mdl_intercept'][fold] = mdl.intercept_
            cv_dict[it]['test_trials'].append(test_idx)
            
        
        # cv = KFold(n_splits=p['nfolds'], shuffle = True)
        # y_pred = cross_val_predict(mdl, X.T, Y, cv = cv)
        y_score = y_pred == Y
        
        # find model accuracy for each time point
        for t in range(max_tpts):
            tr_idx = np.squeeze(np.where(ntpts_per_trial > t))
            if tr_idx.size > 50:
                accuracy_fold[it,t] = np.nanmean(np.array(y_score)[t0_index[tr_idx]+t])
                
    accuracy_sem = np.nan*np.zeros((max_tpts,2))
    accuracy_std = np.nan*np.zeros(max_tpts)
    for t in range(max_tpts):
        accuracy_sem[t,:] = [np.percentile(accuracy_fold[:,t], 2.5),
                            np.percentile(accuracy_fold[:,t], 97.5)]
        accuracy_std[t] = np.std(accuracy_fold[:,t])
    
    
    C = np.mean(mdl.C_)
    mdl = LogisticRegression(
        C = C,
        fit_intercept = True,
        tol = 1e-4,
        penalty = 'l2',
        max_iter = 1e8,
        )
    mdl.fit(X.T, Y)
    
    results = dict()
    results['accuracy'] = np.nanmean(accuracy_fold, axis = 0)  # accuracy(t) from cross-validation across folds
    results['accuracy_sem'] = accuracy_sem  # sem(t) across cross-validation folds
    results['accuracy_std'] = accuracy_std  # std of accuracy(t) across cross-validation folds
    results['cv_summary'] = cv_dict # dict containing test indices and fits across all cross-validation folds and repeats
    results['tpts'] = range(max_tpts)  # which tpts in a trial
    results['mdl_coefs'] = mdl.coef_  # mdl coefs fit to whole data no cross-validation
    results['mdl_intercept'] = mdl.intercept_  # intercept fit to whole data no cross-validation
    results['ref_coef'] = C
    
    return results
    
    
    
        
if __name__ == "__main__":
    
    p = dict()
    p['ratnames'] = SPEC.RATS
    p['regions'] = SPEC.REGIONS
    p['cols'] = SPEC.COLS
    p['fr_thresh'] = 1.0 # firing rate threshold for including neurons
    p['stim_thresh'] = 0.0 # stimulus duration threshold for including trials
    p['align_to'] = ['clicks_on']
    p['align_name'] = ['clickson_masked']
    p['pre_mask'] = [None]
    p['post_mask'] = ['clicks_off']
    p['start_time'] = [0]
    p['end_time'] = [1000] # in ms
    p['binsize'] = 50 # in ms
    p['filter_type'] = 'gaussian'
    p['filter_w'] = 75 # in ms
    p['Cs'] = np.logspace(-7,3,200)  # cross-validation parameter
    p['nfolds'] = 10  # number of folds for cross-validation
    p['n_repeats'] = 10 # number of repeats for cross-validation
    
    
    SAVEDIR = SPEC.RESULTDIR + 'DV_choice_decoding/'
    fname = SAVEDIR + 'params' + datetime()[5:] + ".npy"
    np.save(fname, p)
    
    rat = p['ratnames'][int(sys.argv[1])]
    
    print("\n\n\n\n===== RAT: {} =====".format(rat))
    files, this_datadir = get_sortCells_for_rat(rat, SPEC.DATADIR)
    
    file = files[int(sys.argv[2])]
    print('\n\nProcessing file: {}'.format(file))
    
    fname = SAVEDIR + file[:21] + '_DV_choice_decoding_' + datetime()[5:] + '.npy'
    df_trial, df_cell, _ = load_phys_data_from_Cell(this_datadir + os.sep + file)
    df_trial = df_trial[df_trial['stim_dur_s_actual'] >= p['stim_thresh']].reset_index(drop = True)

    # equalize neurons across regions
    df_cell = df_cell[df_cell['stim_fr'] >= p['fr_thresh']].reset_index(drop = True)
    df_cell = equalize_neurons_across_regions(df_cell, p['regions'])
    
    # equalize trials following left/right correct choice
    df_trial = equalize_trials(df_trial, 'pokedR')
    target = np.array(df_trial['pokedR'], dtype = float)

    summary = dict()
    for align_id, (align, align_name) in enumerate(zip(p['align_to'], p['align_name'])):
        print("\n\nProcessing alignment: {}".format(align_name))
        align_to = p['align_to'][align_id]
        summary[align_name] = dict()
        
        for r, reg in enumerate(p['regions']):
            print("\n\tProcessing region: {}".format(reg))
            X, ntpts_per_trial = get_neural_activity(df_cell, df_trial, reg, p, align_id)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                summary[align_name][reg] = run_logistic_decoding(
                    X,
                    target,
                    ntpts_per_trial,
                    p)
        
    summary['prm'] = dict()
    summary['prm']['filename'] = file
    summary['prm']['ntrials'] = len(df_trial)
    summary['prm']['n_neurons'] = [df_cell['region'].value_counts()['ADS'], df_cell['region'].value_counts()['FOF']]
    np.save(fname, summary)

                
                            
