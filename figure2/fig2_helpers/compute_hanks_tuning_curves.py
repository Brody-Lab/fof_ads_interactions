import sys
sys.path.insert(1, '../../../figure_code/')
from my_imports import *
from tqdm import tqdm
import matplotlib.backends.backend_pdf
from helpers.physdata_preprocessing import load_phys_data_from_Cell
from helpers.rasters_and_psths import *
from scipy.optimize import curve_fit
import hdf5storage

def load_backward_pass(file, plot = True):
    
    accumpath = SPEC.RESULTDIR + 'behavior_data/'
    acc_file = sorted([fn for fn in os.listdir(accumpath) if \
        fn.startswith((file[:-4] + '_rawdata_accum_backward.mat'))])
    acc_raw = np.squeeze(hdf5storage.loadmat(accumpath + acc_file[0])['backward'])
    xc_raw = hdf5storage.loadmat(accumpath + acc_file[0])['xc']
    
    if plot is True:
        plt.figure()
        print(np.shape(acc_raw))
        plt.imshow(acc_raw[0], aspect = "auto");
        plt.title('sample backward pass')
        plt.show()
        
    return acc_raw, np.squeeze(xc_raw)


def coarsen_acc_and_xc(acc_raw, xc, df_trial, num_bins = 13, plotting = False):
    
    xc_mid = len(xc)//2
    new_xc_mid = num_bins//2
    n_rem_bins = new_xc_mid - 1
    
    new_xc = np.nan*np.zeros(num_bins)
    new_xc[new_xc_mid] = xc[xc_mid]
    new_xc[0] = np.mean(xc[:xc_mid - (n_rem_bins*2)])
    new_xc[1:new_xc_mid] = np.mean(np.reshape(xc[xc_mid - (n_rem_bins*2):xc_mid],(n_rem_bins, 2)), axis = 1)
    new_xc[new_xc_mid + 1:] = -1*np.flip(new_xc[:new_xc_mid])
    
    accum = []
    for tr in range(len(df_trial)):
        
        raw = acc_raw[tr]
        this_acc = np.zeros((num_bins, np.shape(raw)[1]))
        this_acc[new_xc_mid,:] = raw[xc_mid,:]

        this_acc[0,:] = np.sum(raw[:xc_mid - (n_rem_bins*2),:], axis = 0)
        first = [(raw[k,:] + raw[k+1,:]) for k in np.arange(xc_mid - (n_rem_bins*2),xc_mid)[::2]]
        this_acc[1:new_xc_mid] = np.array(first)

        this_acc[-1,:] = np.sum(raw[xc_mid + (n_rem_bins*2):,:], axis = 0)
        second = [(raw[k,:] + raw[k+1,:]) for k in np.arange(xc_mid+1, (n_rem_bins*2) + xc_mid)[::2]]
        this_acc[new_xc_mid+1:-1] = np.array(second)

        accum.append(this_acc)
        
        if (plotting == True) & (np.random.rand()<0.01):
            print(tr, df_trial.pokedR[tr], df_trial.click_diff[tr])

            plt.figure(figsize = (10,2))
            edges = np.arange(df_trial.model_start[tr], df_trial.clicks_off[tr], 0.001)
            counts_L, _ = np.histogram(df_trial['leftBups'][tr] , edges)
            counts_R, _ = np.histogram(df_trial['rightBups'][tr] , edges)
            evidence = np.cumsum(counts_R - counts_L)
            plt.plot(-evidence)

            plt.figure(figsize = (10,12))
            plt.imshow(raw, aspect = 'auto', interpolation = 'None', cmap = 'pink')
            ax = plt.gca()
            ax.set_yticks(np.arange(0, 53, 1))
            ax.set_yticklabels(np.arange(0, 53, 1))
            ax.grid(color='red', linestyle='-.', linewidth=0.5)

            plt.figure(figsize = (10,4))
            plt.imshow(this_acc, aspect = 'auto', interpolation = 'None', cmap = 'pink')
            ax = plt.gca()
            ax.set_yticks(np.arange(0, num_bins, 1))
            ax.set_yticklabels(np.arange(0, num_bins, 1))
            ax.grid(color='red', linestyle='-.', linewidth=1)
            plt.show()
        
    return accum, new_xc


def get_neural_data(df_trial, df_cell, cellnum, p, delay, plot = True):
    
    df_trial['align_to'] = df_trial[p['align_to']] + 0.001*delay
    df_trial['post_mask'] = df_trial[p['post_mask']] + 0.001*delay
        
    # plot PSTH for left/right choice
    if plot is True:
        print("Cellnum: ", df_cell.loc[cellnum, 'cell_ID'])
        plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.Spectral(np.linspace(0,1,2)))
        PSTH = make_psth(df_cell.loc[cellnum, 'spiketime_s'], 
                                df_trial,
                                split_by = 'pokedR',
                                align_to = "align_to", 
                                post_mask = "post_mask",
                                window = p['window'],
                                filter_type = p['filter_type'],
                                filter_w = p['filter_w'],
                                binsize = p['binsize'],
                                plot = True)
        plt.title('Aligned_to: ' + p['align_to'] + "+ "+ str(delay) + 'ms' )
        plt.show()

    # finally fetch the data that I am going to use
    PSTH = make_psth(df_cell.loc[cellnum, 'spiketime_s'], 
                            df_trial,
                            split_by = None,
                            align_to = "align_to", 
                            post_mask = "post_mask",
                            window = p['window'],
                            filter_type = p['filter_type'],
                            filter_w = p['filter_w'],
                            binsize = p['binsize'],
                            plot = False)
        
    # return PSTH['trial_fr'][0]/df_cell.loc[cellnum,'fr_stim_onset']
    return PSTH['trial_fr'][0]

    
def closest(lst, K):
    idx = (np.abs(lst - K)).argmin()
    return idx


def make_joint_distribution(accum, fr):
    
    n = np.shape(accum[0])[0]
    nT = [np.shape(accum[tr])[1] for tr in range(len(accum))]

    fr_edges = np.linspace(np.floor(np.nanmin(fr)), np.ceil(np.nanmax(fr)), 50)
    fr_centers = np.diff(fr_edges)[0]/2 + fr_edges[:-1]

    joint_t = np.nan*np.zeros((max(nT), n, len(fr_centers)))
    tr_count = np.zeros(max(nT))

    # compute the joint
    for tr in range(len(accum)):
        for tpt in range(nT[tr]):
            which_fr_bin = closest(fr_centers, fr[tr,tpt])
            if np.any(np.isnan(joint_t[tpt, :, which_fr_bin])):
                joint_t[tpt, :, which_fr_bin] = accum[tr][:,tpt]
            else:
                joint_t[tpt, :, which_fr_bin] += accum[tr][:,tpt]
            tr_count[tpt] += 1

    # normalize by number of trials which contributed to each time point
    for tpt in range(max(nT)):
        joint_t[tpt, :, :] /= tr_count[tpt]

    # normalize the joint
    joint_t /= np.nansum(joint_t)

    return joint_t, fr_centers


def compute_fr_given_at(joint_t, fr_centers, nT, n):
    
    fr_at = np.empty((max(nT), n))
    for tpt in range(max(nT)):
        for idx_n in range(n):
            mat = np.squeeze(joint_t[tpt,idx_n,:])
            if np.nansum(mat) == 0:
                fr_slice = np.nan * np.zeros(np.shape(mat))
            else:
                fr_slice = mat/np.nansum(mat)
            fr_at[tpt,idx_n] = np.nansum(fr_slice * fr_centers) 
            
    return fr_at


def sigmoid(x, x0, k):
    y =  1 / (1 + np.exp(-k*(x-x0)))
    return (y)


def sigmoid_4(x, x0, k, a, b):
    y =  a + (b/ (1 + np.exp(-k*(x-x0))))
    return (y)


def compute_tuning_curve(fr_at, fr, df_trial, p, xc, pdf = None, title = ''):
    
    n = np.shape(fr_at)[1]

    plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.Spectral(np.linspace(0,1,n)))
    colors = plt.cm.Spectral(np.linspace(0,1,n))[:,:3]
    window = adjust_window(p['window'], p['binsize'])
    edges = np.arange(window[0], window[1] + p['binsize'], p['binsize'])*0.001 # convert to s 

    fig, axs = plt.subplots(3,2, figsize = (10,10), gridspec_kw = {'height_ratios':[2,3,2]})
    t0 = int(p['t0_ms']/p['binsize'])
    t1 = int(p['t1_ms']/p['binsize'])    
    fr = fr[:,t0:t1]
    Δ_fr_at = fr_at[t0:t1,:] 

    
    choice = split_trials(df_trial, 'pokedR')
    psth_left = np.nanmean(fr[choice[0]], axis = 0)
    ntr = [sum(~np.isnan(fr[choice[0],t])) for t in range(np.shape(fr[choice[0],:])[1])]
    psth_left_sem = np.sqrt(np.nanvar(fr[choice[0]], axis = 0))/np.sqrt(ntr)
    psth_right = np.nanmean(fr[choice[1]], axis = 0)
    ntr = [sum(~np.isnan(fr[choice[1],t])) for t in range(np.shape(fr[choice[1],:])[1])]
    psth_right_sem = np.sqrt(np.nanvar(fr[choice[1]], axis = 0))/np.sqrt(ntr)

    axs[0,0].plot(edges[t0:t1], psth_left, c = colors[1], label = "Choice L")
    axs[0,0].plot(edges[t0:t1], psth_right, c = colors[-2], label = "Choice R")
    axs[0,0].legend()
    axs[0,0].fill_between(edges[t0:t1], 
                          psth_left - psth_left_sem,
                          psth_left + psth_left_sem, alpha = 0.25, color = colors[1])
    axs[0,0].fill_between(edges[t0:t1], 
                          psth_right - psth_right_sem,
                          psth_right + psth_right_sem, alpha = 0.25, color = colors[-2])
    axs[0,0].set_xlabel('Time from stimulus onset[s]')
    axs[0,0].set_ylabel('Firing rate [spikes/s]')
    axs[0,0].set_ylim([np.min(Δ_fr_at), np.max(Δ_fr_at)])
    axs[0,0].set_title('Choice PSTH')

    im = axs[0,1].imshow(Δ_fr_at.T, aspect = 'auto', cmap = 'pink', interpolation = 'None') 
    axs[0,1].axhline(n//2, c = 'k')
    clb = plt.colorbar(im, ax=axs[0, 1])
    clb.ax.set_title('Firing rate')
    axs[0,1].set_yticks(np.arange(0, n,2))
    axs[0,1].set_yticklabels(np.round(xc[np.arange(0, n, 2)]))
    axs[0,1].set_ylabel('Accumulator bins')
    axs[0,1].set_xticks(np.arange(0, t1-t0, 100))
    axs[0,1].set_xticklabels(edges[np.arange(100, 100+t1-t0, 100)])
    axs[0,1].set_xlabel('Time from stimulus onset[s]')
    
    for i in range(1,n-1):
        axs[1,0].plot(edges[t0:t1], fr_at[t0:t1,i], color = colors[i]); 
    axs[1,0].set_xlabel('Time from stimulus onset [s]')
    axs[1,0].set_ylabel('Firing rate')
    
    # Tuning curve computation - make this a different function (!!)
    mean_tuning = np.mean(Δ_fr_at, axis = 0)
    std_tuning = np.std(Δ_fr_at, axis = 0)
    mod_range = np.round(np.max(mean_tuning[1:-1]) - np.min(mean_tuning[1:-1]),2)
    
    mean_tuning -= np.min(mean_tuning[1:-1])
    norm_mean = np.max(mean_tuning[1:-1])
    mean_tuning /= norm_mean
    std_tuning /= norm_mean
#     popt, pcov = curve_fit(sigmoid, xc[1:-1], mean_tuning[1:-1], 
#                        method = "trf", 
#                        maxfev = 30000,
#                        bounds = ([-3., -np.inf],[3., np.inf]))
    
    popt, pcov = curve_fit(
        sigmoid_4, xc[1:-1],
        mean_tuning[1:-1], 
        method = "trf", 
        maxfev = 30000,
        bounds = ([-np.inf, 0.0, 0, 0],[np.inf, 2.5, 1, 2]),
        p0 = [0, 1, 0, 1])
    
    axs[1,1].scatter(xc, mean_tuning, s = 20, color = colors, zorder = 3)
    axs[1,1].errorbar(
        xc, mean_tuning, 
        yerr= std_tuning, 
        ls = "none", 
        c = 'k', 
        lw = 0.5, 
        zorder = 1)
    axs[1,1].plot(xc[1:-1], sigmoid_4(xc[1:-1], *popt),
                c = 'k', ls = '-', lw = 0.8, zorder = 2)
    axs[1,1].set_xlabel('Accumulator value')
    axs[1,1].set_ylabel('Δ norm firing rate')
    axs[1,1].set_title('FR modulation (pre normalization): ' + str(mod_range))
    
    
    # rank 1 approximation of the tuning curve
    u,s,vh = np.linalg.svd(Δ_fr_at , full_matrices = False)
    u = -1*u
    vh = -1*vh
    reconstruction = s[0]*np.outer(u[:,0], vh[0,:])
    for i in range(1,n-1):
        axs[2,0].plot(edges[t0:t1], reconstruction[:,i], color = colors[i])
    var_explained = np.round(s[0]**2/np.sum(s**2),2)
    axs[2,0].set_title("Rank 1 reconstruction\nVariance captured = " + str(var_explained))
    
    v = vh[0,:]
    v -= np.min(v[1:-1])
    norm = np.max(v[1:-1])
    v /= norm

    popt_rank1, pcov = curve_fit(
        sigmoid_4, xc[1:-1], v[1:-1], 
        method = "trf", 
        maxfev = 30000,
        bounds = ([-np.inf, 0.0, 0, 0],[np.inf, 2.5, 1, 2]))
    axs[2,1].scatter(xc, v, s = 20, color = colors, zorder = 3)
    axs[2,1].plot(xc[1:-1], sigmoid_4(xc[1:-1], *popt_rank1), c = 'k', ls = '-', lw = 0.8, zorder = 2)
    axs[2,1].set_title('Rank 1 tuning curve')
    
    fig.suptitle(title, fontsize = 14)
    
    sns.despine()
    plt.tight_layout()
#     plt.show()
    
    if pdf is not None:
        pdf.savefig(fig)
    plt.close()
    
    return mean_tuning, std_tuning, popt, popt_rank1, var_explained, mod_range


def plot_joint_and_marginals_over_time(joint_t, fr_edges, xc, nT):

    for tr in np.arange(max(nT))[::4]:
        fig, axs = plt.subplots(1,3, figsize = (6,2))    
        axs[0].imshow(np.squeeze(joint_t[tr,:,:]).T, aspect = "auto")
        axs[1].plot(fr_edges,np.mean(joint_t[tr,:,:], axis = 0))
        axs[2].plot(xc,np.mean(joint_t[tr,:,:], axis = 1))

        sns.despine()
        plt.tight_layout()
        plt.show()
        
        
        

if __name__ == "__main__":
    
    p = dict()

    p['ratnames'] = SPEC.RATS
    p['regions'] = SPEC.REGIONS
    p['fr_thresh'] = 1.0
    p['side_pref_thresh'] = 0.05
    p['auc_thresh'] = 0.075
    p['correct_only'] = False
    p['cols'] = SPEC.COLS
    p['align_to'] = "model_start"
    p['post_mask'] = "clicks_off"
    p['window'] = [0, 1500]
    p['binsize'] = 1
    p['filter_type'] = "gaussian"
    p['filter_w'] = 75
    p['neural_delay'] = dict()
    p['neural_delay']["FOF"] = 100 # in ms
    p['neural_delay']["ADS"] = 100 # in ms
    p['t0_ms'] = 150
    p['t1_ms'] = 500

    df_sd = pd.DataFrame()
    df_sd["rat"] = ""
    df_sd["region"] = ""
    df_sd["filename"] = ""
    df_sd["cell_ID"] = ""
    df_sd["stim_fr"] = ""
    df_sd["side_pref"] = "" 
    df_sd["side_pref_p"] = "" 
    df_sd["auc"] = "" 

    df_sd["mean_tuning"] = ""
    df_sd["std_tuning"] = ""
    df_sd["xc"] = ""
    df_sd["fr_mod_range"] = ""
    df_sd["tun_slope"] = ""
    df_sd["tun_bias"] = ""
    df_sd["tun_slope_rank1"] = ""
    df_sd["tun_bias_rank1"] = ""
    df_sd["var_exp"] = ""

    data_dict = dict()
    data_dict['fr_at'] = []

    savepath = SPEC.RESULTDIR + 'hanks_tuning_curves/'
    fmarker = "hanks_tuning_curve_all_sig_neurons.pdf"
    
        
    for rat in p['ratnames']:
        
        print("\n\n\n\n====== RAT: ", rat, " ======")
        files, this_datadir = get_sortCells_for_rat(rat, SPEC.DATADIR)

        for file in files:
            
            print("\n\nProcessing file: ", file)   
            pdf = matplotlib.backends.backend_pdf.PdfPages(savepath + file[:19] \
                                                        + "hanks_tuning_summary" + fmarker)
            df_trial, df_cell, _ = load_phys_data_from_Cell(this_datadir + os.sep + file)

            # df_trial_modifications
            df_trial['model_start'] = df_trial['clicks_on'] - (df_trial['stim_dur_s_theoretical'] - df_trial['stim_dur_s_actual'])

            
            # df_cell modifications
            df_cell = df_cell[np.abs(df_cell['auc']-0.5) > p['auc_thresh']]
            df_cell = df_cell[df_cell['side_pref_p'] <= p['side_pref_thresh']].reset_index()
            
            # compute mean fr at the beginning of each trial
            for c in range(len(df_cell)):
                # mean firing rate at stim start for normalization
                PSTH_mean = make_psth(df_cell.loc[c, 'spiketime_s'], 
                                df_trial,
                                split_by = None,
                                align_to = "clicks_on", 
                                post_mask = None,
                                window = [-50,50],
                                filter_type = p['filter_type'],
                                filter_w = p['filter_w'],
                                binsize = 1,
                                plot = False)['data'][0]['mean']
                df_cell.loc[c, 'fr_stim_onset'] = np.mean(PSTH_mean) 
            df_cell = df_cell[df_cell['fr_stim_onset'] >= p['fr_thresh']].reset_index(drop = True)
            


            # load and rebin backward pass:
            # accum is len(xc)xt (in ms) dim and xc are the bin centers 
            acc_raw, xc_raw = load_backward_pass(file, plot = False)
            if p['correct_only'] == True:
                is_hit = np.where(df_trial.is_hit == 1)[0]
                df_trial = df_trial.loc[is_hit].reset_index(drop = True)
                acc_raw = acc_raw[is_hit]
            accum, xc = coarsen_acc_and_xc(acc_raw, xc_raw, df_trial, plotting = False)
            assert len(accum) == len(df_trial)

            n = len(xc)
            nT = [np.shape(accum[tr])[1] for tr in range(len(accum))]

            for cellnum in tqdm(range(len(df_cell))):

                reg = df_cell.region[cellnum]
                delay = p['neural_delay'][reg]
                fr = get_neural_data(df_trial, df_cell, cellnum, p, delay, plot = False)

                joint_t, fr_centers = make_joint_distribution(accum, fr)
                fr_at  = compute_fr_given_at(joint_t, fr_centers, nT, n)
                
                if df_cell.loc[cellnum, 'side_pref'] == 0:
                    fr_at = np.flip(fr_at, axis = 1)
                    
                
                data_dict['fr_at'].append(fr_at[p['t0_ms']:p['t1_ms'],:])

                title =  'Cell number: ' + df_cell.loc[cellnum, 'cell_ID'] \
                + ' (' + reg + ')' + ' auc: ' + str(np.round(df_cell.loc[cellnum, 'auc'],2))
                mean_tun, std_tun, popt, popt_rank1, var_exp, fr_mod_range = compute_tuning_curve(
                    fr_at, 
                    fr, 
                    df_trial, 
                    p, 
                    xc, 
                    pdf, 
                    title = title) 

                idx = len(df_sd)
                df_sd.loc[idx, "rat"] = rat
                df_sd.loc[idx, "region"] = reg
                df_sd.loc[idx, "filename"] = file
                df_sd.loc[idx, "cell_ID"] = df_cell.loc[cellnum, 'cell_ID']
                df_sd.loc[idx, "stim_fr"] = df_cell.loc[cellnum, "stim_fr"]
                df_sd.loc[idx, "side_pref"] = df_cell.loc[cellnum, "side_pref"]
                df_sd.loc[idx, "side_pref_p"] = df_cell.loc[cellnum, "side_pref_p"] 
                df_sd.loc[idx, "auc"] = df_cell.loc[cellnum, "auc"] 

                df_sd.loc[idx, "xc"] = xc
                df_sd.loc[idx, "mean_tuning"] = mean_tun
                df_sd.loc[idx, "std_tuning"] = std_tun
                df_sd.loc[idx, "tun_slope"] = popt[1]*popt[3]/4
                df_sd.loc[idx, "tun_bias"] = popt[0]
                # df_sd.loc[idx, "tun_slope_rank1"] = popt_rank1[1]*popt_rank1[3]/4
                # df_sd.loc[idx, "tun_bias_rank1"] = popt_rank1[0]
                df_sd.loc[idx, "var_exp"] = var_exp
                df_sd.loc[idx, "fr_mod_range"] = fr_mod_range


            pdf.close()


    df_sd['fr_thresh'] = p['fr_thresh']
    df_sd['side_pref_thresh'] = p['side_pref_thresh']
    df_sd['auc_thresh'] = p['auc_thresh']
    df_sd['correct_only'] = p['correct_only']
    df_sd['align_to'] = p['align_to']
    df_sd['post_mask'] = p['post_mask']
    df_sd['binsize'] = p['binsize']
    df_sd['filter_type'] = p['filter_type']
    df_sd['filter_w'] = p['filter_w']
    df_sd['window0'] = p['window'][0]
    df_sd['window1'] = p['window'][1]

    for reg in p['regions']:
        df_sd['neural_delay_'+reg] = p['neural_delay'][reg]
    df_sd['t0_ms'] = p['t0_ms']
    df_sd['t1_ms'] = p['t1_ms']
    
    
    # save to CSV
    df_sd.to_csv(savepath + "hanks_tuning_curve_analysis_150_to_500.csv")

    
    