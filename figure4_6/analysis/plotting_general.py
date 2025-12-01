import numpy as np
import pandas as pd
import seaborn as sns
import random, os
import itertools

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from scipy.optimize import curve_fit
from scipy.special import expit

import sys
# sys.path.append('/home/dikshag/pbups_ephys_analysis/pbups_phys/')
# from phys_helpers import split_trials

# some default plotting settinfs
import seaborn as sns
sns.set_style("white")
sns.set_theme(context='paper', 
              style='ticks',  
              font='Helvetica', 
              font_scale=1.3,  
              rc={"axes.titlesize": 13})
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['axes.unicode_minus']=False


def split_trials(df, split_by):
    '''Computes indices corresponding to unique values of a variable in a dataframe.
    Args:
        df : The Dataframe you are seeking to split. Must contain `split_by` as a column
        split_by (str): Column name in the dataframe `df`, based on unique values of which
            indices will be grouped. can be None, if no splitting is desired
    Returns:
        dict: keys are the unique values of `split_by` and entries whithin keys are indices 
        if `split_by` is None, then all the indices are returned with key `0`
    '''
    trials = dict()
    if split_by is not None:
        conditions = np.sort(df[split_by].unique())
        trials = {cond : np.where(df[split_by] == cond)[0] for cond in conditions}
    else:
        trials[0] = np.arange(len(df))
    return trials

  
def plot_weights(weights,title="", ax = None):
    
    cmap = plt.set_cmap('RdBu_r')
    if ax == None:
        img = plt.matshow(weights, norm=Normalize(vmin=-.2, vmax=.2))
        plt.title(title)
        plt.colorbar();      
    else:
        img = ax.matshow(weights, norm=Normalize(vmin=-.2, vmax=.2))
        ax.set_title(title)
#         plt.colorbar(img, orientation='vertical')
    
    
    
    
def plot_psych(df_trial, var, ax, legend = None, num_bins = 20, color = 'k', ls = '-'):
            
    def psych_func(x, s, b, g, l):
        '''sigmoid for curve fitting
        '''
        return g + l*expit(s*(x+b))
    
    
    Δcuts, bins = pd.cut(df_trial['Δclicks'],bins = num_bins, retbins = True)
    bin_centers = bins[:-1] + np.diff(bins)[0]/2
    
    # first for target
    try:
        popt, pcov = curve_fit(psych_func, 
                               np.array(df_trial['Δclicks']).astype('float32'),
                               np.array(df_trial[var]).astype('float32'),
                               maxfev = 20000)
        psych = psych_func(np.linspace(-40,40, 81), *popt)
        ax.plot(np.linspace(-40,40, 81), psych, label = legend, color = color, ls = ls)
    except:
        leg = 4
    mean_accuracy = df_trial.groupby(Δcuts)[var].mean()
    ax.set_ylabel('Fraction rightward choices')
    ax.set_xlabel('(#R - #L) clicks')
    ax.set_ylim([0,1])
    if legend is not None:
        ax.legend(frameon = False)
    sns.despine()

    
    


def plot_trial(axs, data_dict, df_trial, tr, dt, legend = False):
    
    x = data_dict['x']
    y = data_dict['y']
    output = data_dict['output']
    t = range(0, len(x[0,:,:])*dt,dt)
    
    legend = ["L clicks", "R clicks", "Lapse", "History"][:x.shape[1]]

    # inputs
    axs[0].plot(t, x[tr,:,:], lw = 1)
    axs[0].axvline(1500, c = 'k', ls = '--')
    axs[0].axvline(df_trial.loc[tr, 'onset_time'], c= 'k', ls = ':')
    axs[0].set_ylabel("Input Magnitude")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylim([-5,5])
    if legend == True:
        axs[0].legend(legend, ncol = 2);
        axs[0].set_title("Input Data")

    # accmulator
    axs[1].plot(t,df_trial.loc[tr, 'accumulator'], c = 'grey', lw = 1.5)
    axs[1].axhline(0, c= 'k', ls = ':')
    axs[1].axvline(df_trial.loc[tr, 'onset_time'], c= 'k', ls = ':')
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylim([-15,15])
    if legend == True:
        axs[1].set_title("Accumulator")
 
    # target / provisional choice
    axs[2].plot(t,y[tr,:,:])
    axs[2].axvline(df_trial.loc[tr, 'onset_time'], c= 'k', ls = ':')
    axs[2].axvline(1500, c = 'k', ls = '--')
    axs[2].set_ylabel("Target output")
    axs[2].set_xlabel("Time (ms)")
    axs[2].set_ylim([-1.5,1.5])
    if legend == True:
        axs[2].legend(["L", "R"]);
        axs[2].set_title("Target Output")
        
    # network output
    axs[3].plot(t,output[tr,:,:])
    axs[3].axhline(-1, c= 'k', ls = ':')
    axs[3].axhline(1., c= 'k', ls = ':')
    axs[2].axvline(1500, c = 'k', ls = '--')
    axs[3].axvline(df_trial.loc[tr, 'onset_time'], c= 'k', ls = ':')
    axs[3].set_ylabel("Activity of Output Unit")
    axs[3].set_xlabel("Time (ms)")
    axs[3].set_ylim([-2,2.])
    if legend == True:
        axs[3].set_title("Output")
        axs[3].legend(["L output", "R output"]);

    sns.despine()

    
    
    
    
def plot_trial_activty(activity, trialnum, dt):
    fig, axs = plt.subplots(1,3, figsize = (12,4), sharey= True)

    t = range(0, np.shape(activity)[2]*dt,dt)
    axs[0].plot(t, activity[:100, trialnum, :].T)
    axs[0].set_ylabel("State Variable Value")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_title("FOF")

    axs[1].plot(t,activity[200:,trialnum, :].T)
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_title("ADS")

    axs[2].plot(t,activity[100:200,trialnum, :].T)
    axs[2].set_xlabel("Time (ms)")
    axs[2].set_title("in b/w")
    
    
    
    
    
    
    
def make_RNN_PSTHs(df_trial, activity, dt, split_by = 'choice', align_to = None, num_per_col = 10):
     
    from scipy.ndimage import gaussian_filter
    import warnings
    
    def get_color_iter(splits):
        if len(list(splits)) == 2:
            colors = sns.color_palette("BuGn", 8)
            colors = np.vstack((colors[0], colors[-1]))
            palette = iter(colors)
        else:
            palette = iter(sns.diverging_palette(43, 144, 
                                                 s=90,
                                                 l=50, 
                                                 sep = 1, 
                                                 n = len(list(splits))))
        return palette
    
    
    splits = split_trials(df_trial, split_by)
    if split_by == "difficulty":
        splits = {key: splits[key] for key in ['R easy', 'R hard', 'L hard', 'L easy']}

    num_cells = np.shape(activity)[0]
    nrows = np.ceil(num_cells/num_per_col).astype(int)
    t = range(0, np.shape(activity)[2]*dt,dt)
    fig,axs = plt.subplots(nrows, num_per_col, 
                           figsize = (num_per_col*2, nrows*2), 
                           sharex = True,
                           sharey = False)
    
    # I am only setting it up for two alignments - what other alignments are there?
    if align_to == None:
        
        for i, ax in zip(range(num_cells), axs.ravel()):
            palette = get_color_iter(splits)
            for k in list(splits):
                color = next(palette)
                mean_activity = np.mean(activity[i][splits[k],:], axis = 0)
                ax.plot(t, gaussian_filter(mean_activity,5), c = color, lw = 2)
                
    elif align_to == 'stim_onset':
        
        stim_on = (df_trial['onset_time']/dt).astype(int) 
        stim_off = ((df_trial['onset_time']+df_trial['stim_duration'])/dt).astype(int) 
        ntrials = len(df_trial)

        for i, ax in zip(range(num_cells), axs.ravel()):
            
            # align and pad with nans
            fr_cpoke_align = [activity[i][tr, stim_on[tr]:stim_off[tr]] for tr in range(ntrials)]
            fr_cpoke_align = np.array(list(itertools.zip_longest(*fr_cpoke_align,fillvalue=np.nan))).T
            
            palette = get_color_iter(splits)
            for k in list(splits):
                color = next(palette)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_activity = np.nanmean(fr_cpoke_align[splits[k],:], axis = 0)
                    ax.plot(np.array(t[:len(mean_activity)])/1000,
                            gaussian_filter(mean_activity,3), 
                            c = color,
                            label = k)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, 
               labels, 
               frameon = False, 
               ncol = 1, 
               handlelength = 1.0,
               handleheight = 2.0,
               bbox_to_anchor = (1.25, 0.85, 0, 0),
               fontsize = 12)
    fig.text(0.5, 0.0, 'Time from clicks on [s]', ha='center', fontsize = 15)
    fig.text(-0.01, 0.5, 'Firing rate [au]', va='center', rotation='vertical', fontsize = 15)
    
    plt.tight_layout()
    sns.despine()
    
    
    
    
def plot_inactivation_summary(data, gain = None):
        
    perturb_grp = list(data)
    n_perturb_grp = len(perturb_grp)
    
    if gain is None:
        gain = list(data[perturb_grp[0]])
    n_gain = len(gain)
        
    cols = {'first_half': [i/255 for i in [250,207,119]],
            'second_half': [i/255 for i in [136,163,113]]}
    
    fig, axs = plt.subplots(n_gain, 
                            n_perturb_grp, 
                            figsize = (2.1*n_perturb_grp, 3*n_gain),
                            sharex = True, 
                            sharey = True)
    
    
    def plot_this_scatter(this_data, this_ax):

        npts = len(this_data['first_half'])
        epochs = list(this_data)
        for (epoch, offset) in zip(epochs, [1., 1.7]):
            this_ax.scatter(offset*np.ones(npts)+np.random.rand(npts)*0.2, 
                            this_data[epoch], 
                            alpha = 0.2, 
                            color= cols[epoch])
            # this_ax.plot([offset, offset + 0.2], 
            #              [np.mean(this_data[epoch]), np.mean(this_data[epoch])], 
            #              lw = 5,
            #              color= cols[epoch])
            this_ax.errorbar(offset+0.1, 
                             np.mean(this_data[epoch]), 
                             yerr=np.std(this_data[epoch]), 
                             fmt='o',
                             color = 'k')
            this_ax.scatter(offset + 0.1,
                            np.mean(this_data[epoch]),
                            60,
                            color = cols[epoch],
                            edgecolor = 'k',
                            zorder = 100)
            # this_ax.plot([offset + 0.1, offset + 0.1], 
            #              [np.mean(this_data[epoch]) - np.std(this_data[epoch]),
            #              np.mean(this_data[epoch]) + np.std(this_data[epoch])], 
            #              lw = 3,
            #              color= cols[epoch])
  
        this_ax.set_xlim([0.7, 2.2])
        this_ax.axhline(0, c = 'k', ls = '--')
        this_ax.set_xticks([1.1,1.8], labels = ['1$^{st}$ half', '2$^{nd}$ half'], fontsize = 12)
        for t, tick in enumerate(this_ax.get_xticklabels()):
            tick.set_bbox(dict(facecolor=cols[epochs[t]], edgecolor=None))
        
    
    def setylabel(this_ax):
        if 'bi' in this_grp:
            this_ax.set_ylabel('Accuracy (%) \n (control - inactivation)', fontsize = 14)
        else:
            this_ax.set_ylabel("Mean ipsilateral bias \n compared to control trials $\pm$ SD", fontsize = 14)

    
    if np.ndim(axs) == 0:
        this_grp = perturb_grp[0]
        this_gain = gain[0]
        plot_this_scatter(data[this_grp][this_gain], axs)
        axs.set_title('FOF-ADS proj' if this_grp == 'proj' else this_grp, fontsize = 14)
        setylabel(axs)
        
    else:
        ax = axs.ravel()
        for p, this_grp in enumerate(perturb_grp):
            for g, this_gain in enumerate(gain):
                plot_this_scatter(data[this_grp][this_gain], ax[n_perturb_grp*g + p])
                if g == 0:
                    ax[n_perturb_grp*g + p].set_title('FOF-ADS proj' if this_grp == 'proj' else this_grp, fontsize = 14)
                if p == 0:
                    setylabel(ax[n_perturb_grp*g + p])
                    
        plt.tight_layout()

    print('RNN inactivations (gain = ' + str(gain) + ')')
    sns.despine()
    
    
    
    

def plot_mean_pca_proj_3d(split, pca_proj, ax, cmaplist, backcol = 'gray', split_by = 'Correct choice: '):
    
    if backcol != 'gray':
        addlabel = ' (inactivation)'
    else:
        addlabel = " (control)"
    
    for s,c in enumerate(list(split)):
        
        # plot trajectory
        ax.plot3D(np.mean(pca_proj[0,split[c], :], axis = 0), 
                  np.mean(pca_proj[1,split[c], :], axis = 0),
                  np.mean(pca_proj[2,split[c], :], axis = 0), 
                  backcol, 
                  label = addlabel,
                  linewidth = 3.5)
     
        # scatter trajectory
        ax.scatter3D(np.mean(pca_proj[0,split[c], :], axis = 0), 
                     np.mean(pca_proj[1,split[c], :], axis = 0),
                     np.mean(pca_proj[2,split[c], :], axis = 0), 
                     s = 5,
                     depthshade = False,  
                     c=np.arange(np.shape(pca_proj)[2]), linewidth = 0.5,
                     cmap=cmaplist[s])
  
        # scatter initial point
        ax.scatter3D(np.mean(pca_proj[0,split[c], :], axis = 0)[0], 
                     np.mean(pca_proj[1,split[c], :], axis = 0)[0],
                     np.mean(pca_proj[2,split[c], :], axis = 0)[0], 
                     s = 120, marker = '^', c = 'orange', 
                     edgecolor = backcol,
                     zorder = 1000,
                     label = 't=0' + addlabel)
        

        # scatter 500ms in
        ax.scatter3D(np.mean(pca_proj[0,split[c], :], axis = 0)[49], 
                     np.mean(pca_proj[1,split[c], :], axis = 0)[49],
                     np.mean(pca_proj[2,split[c], :], axis = 0)[49], 
                     s = 120, marker = 's', c = 'k', 
                     edgecolor = backcol, 
                     zorder = 1000,
                     label = 't=500ms' + addlabel)
        
        
        
        # scatter 1000ms in
        ax.scatter3D(np.mean(pca_proj[0,split[c], :], axis = 0)[99], 
                     np.mean(pca_proj[1,split[c], :], axis = 0)[99],
                     np.mean(pca_proj[2,split[c], :], axis = 0)[99], 
                     s = 120, marker = 's', c = 'y', 
                     edgecolor = backcol, 
                     zorder = 1000,
                     label = 't=1000ms' + addlabel)
    
                   
    ax.set_xlabel('activity along PC1') #, fontsize = 9)
    ax.set_ylabel('activity along PC2') #, fontsize = 9)
    ax.set_zlabel('activity along PC3') #, fontsize = 9)
    
    

    
def plot_mean_pca_proj_2d(split, pca_proj, ax, cmaplist, backcol = 'gray', split_by = 'Correct choice: '):
    
    if backcol != 'gray':
        addlabel = ' (inactivation)'
    else:
        addlabel = " (control)"
    
    for s,c in enumerate(list(split)):
        
        # plot trajectory
        ax.plot(np.mean(pca_proj[0,split[c], :], axis = 0), 
                  np.mean(pca_proj[1,split[c], :], axis = 0),
                  backcol, 
                  label = addlabel,
                  linewidth = 0.8)
     
        # scatter trajectory
        ax.scatter(np.mean(pca_proj[0,split[c], :], axis = 0), 
                     np.mean(pca_proj[1,split[c], :], axis = 0),
                     s = 8,
                     c=np.arange(np.shape(pca_proj)[2]), linewidth = 0.5,
                     cmap=cmaplist[s])
  
        # scatter initial point
        ax.scatter(np.mean(pca_proj[0,split[c], :], axis = 0)[0], 
                     np.mean(pca_proj[1,split[c], :], axis = 0)[0],
                     s = 20, marker = '^', c = 'k', 
                     edgecolor = backcol,
                     label = 't=0' + addlabel)
        
        # scatter 500ms in
        ax.scatter(np.mean(pca_proj[0,split[c], :], axis = 0)[49], 
                     np.mean(pca_proj[1,split[c], :], axis = 0)[49],
                     s = 40, marker = 's', c = 'k', 
                     edgecolor = backcol, 
                     label = 't=500ms' + addlabel)
        
        # scatter 1000ms in
        ax.scatter(np.mean(pca_proj[0,split[c], :], axis = 0)[99], 
                     np.mean(pca_proj[1,split[c], :], axis = 0)[99],
                     s = 40, marker = 's', c = 'y', 
                     edgecolor = backcol, 
                     label = 't=1000ms' + addlabel)
    
                   
    ax.set_xlabel('activity along PC0', fontsize = 9)
    ax.set_ylabel('activity along PC1', fontsize = 9)
    
    
