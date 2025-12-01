import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from scipy.optimize import curve_fit
from scipy.special import expit
import sys
import pickle

# plt.rcParams['axes.prop_cycle'] = plt.cycler("color", plt.cm.Spectral(np.linspace(0,1,8)))
# plt.rcParams['figure.figsize'] = [4, 3]
# plt.rcParams["font.family"] = "Helvetica"

sys.path.insert(1, '../../../figure_code/')
from my_imports import *
from helpers.phys_helpers import flatten_list, xcorr, savethisfig
from helpers.rasters_and_psths import make_psth, get_neural_activity
from figure2.fig2_helpers.logisticdecoding import run_logistic_decoding


def nanpad_ragged_sequence(ragged_seq):
    
    max_length = max(len(subseq) for subseq in ragged_seq)

    # Create a new list with padded sequences
    padded_seq = [list(subseq) + [np.nan] * (max_length - len(subseq)) for subseq in ragged_seq]

    # Convert the list to a NumPy array
    array_with_nan_padding = np.array(padded_seq)
    
    return array_with_nan_padding


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


def psych_func(x, s, b, g, l):
    '''
    sigmoid for curve fitting
    '''
    return g + l * expit(s * (x+b))


def make_dataframes(task_data, ns, spikes):

    df_trial = pd.DataFrame(task_data)
    df_trial = df_trial.rename(columns = {"choice": "pokedR", "history_bias": "history", "T": "stim_dur"})
    df_trial["clicks_on"] = np.arange(len(df_trial))
    df_trial["clicks_off"] = df_trial["clicks_on"] + df_trial["stim_dur"]
    
    df_cell = pd.DataFrame()
    for i in range(2*ns.N_neurons_per_region):
        df_cell.loc[i, 'region'] = "A" if i < ns.N_neurons_per_region else "B"
        df_cell.loc[i, 'spiketime_s'] = ''
        spiketime_s = []
        for tr in range(len(df_trial)):
            this_spikes = np.where(spikes[tr,:,i] > 0)[0]*ns.dt
            if this_spikes.size > 0:
                spiketime_s.append([np.round(s + df_trial.loc[tr, 'clicks_on'],3) for s in this_spikes])
        df_cell['spiketime_s'][i] = flatten_list(spiketime_s)

    return df_trial, df_cell


def crosscorrelate_DVs(DV, prm, shuffle = False):
    
    regions = list(DV)
    num_tpts = DV[regions[0]].shape[1]
    num_trials = DV[regions[0]].shape[0]
    DV_crosscoef = np.nan*np.zeros((num_trials, 2*num_tpts-1))
    
    # set shuffle
    if shuffle == True:
        l0 = np.random.permutation(num_trials)
    else:
        l0 = range(num_trials)
        
    # now cross-correlate
    for tr, tr0 in enumerate(l0):
        nonnan = ~np.isnan(DV[list(DV)[0]][tr, :])
        lags, values = xcorr(
            DV[list(DV)[1]][tr, nonnan],
            DV[list(DV)[0]][tr0, nonnan],
            detrend = False,
            scale = 'unbiased'
        )
        DV_crosscoef[tr, lags + num_tpts -1] = values
        
    # create summary
    DV_dict = dict()
    # DV_dict['DV_crosscoef'] = DV_crosscoef
    DV_dict['lags'] = np.arange(-num_tpts+1, num_tpts) * prm['binsize'] * 1e-3
    DV_dict['mean'] = np.nanmean(DV_crosscoef, axis = 0)
    DV_dict['sem'] = np.nanstd(DV_crosscoef, axis = 0)/np.sqrt(DV_crosscoef.shape[0])
    idx = np.abs(DV_dict['lags']) < 0.1
    peak_idx = np.argmax(np.nanmean(DV_crosscoef[:,idx], axis = 0))
    DV_dict['peak'] = DV_dict['lags'][idx][peak_idx]
    
    return DV_dict

# plot DV crosscorrelation
def plot_cc(this_cc, color, ax, label, alpha = 0.5):
    ax.plot(this_cc['lags'], this_cc['mean'], c= color, label = label)
    ax.fill_between(this_cc['lags'],  
                this_cc['mean'] - this_cc['sem'], 
                this_cc['mean'] + this_cc['sem'], 
                alpha = alpha, 
                color = color,
                edgecolor = None)


def plot_cross_corr_metrics(DV_summary, DV, ns, summary, df_trial, latents):
    
    fig_DV = plt.figure(constrained_layout = True, figsize = (10,10))
    gs = fig_DV.add_gridspec(5,4)
    fig_DV.suptitle('DV cross correlation summary', fontsize = 16)
            
    regions = list(DV_summary['mdl_coefs'])
    colors = ['b', 'g']
    # plot inferred DV axis compared to emissions
    for r, reg in enumerate(regions):
        ax0 = fig_DV.add_subplot(gs[0, r])
        this_modelcoef = DV_summary['mdl_coefs'][reg].T
        this_emissions = ns.C[r*nn:(r+1)*nn,r*ns.N_latents_per_region]
        ax0.plot(
            this_modelcoef/np.linalg.norm(this_modelcoef),
            label = 'DV axis',
            c = 'k')
        ax0.plot(
            this_emissions/np.linalg.norm(this_emissions), 
            c= 'r', 
            label = 'weights')
        this_corr = np.round(np.corrcoef(this_modelcoef.T, this_emissions)[0,1],2)
        ax0.legend(frameon = False)
        ax0.set_xlabel('Neuron number')
        ax0.set_title('REG: {} | Corr: {}'.format(reg, this_corr))
        
    # plot decoding accuracy
    ax0 = fig_DV.add_subplot(gs[0,2])
    for r, reg in enumerate(regions):
        ax0.plot(
            summary[reg]['accuracy'], 
            label = "reg" + reg,
            color = colors[r])
    ax0.legend()
    ax0.set_ylim([0.5, 1.0])
    ax0.set_title("Decoding accuracy")
        
    # next plot a couple of examples 
    tr = np.random.choice(nt)
    ax0 = fig_DV.add_subplot(gs[0,3])
    for r, reg in enumerate(regions):
        ax0.plot(
            DV[reg][tr,:], 
            label = "reg" + reg,
            color = colors[r])
    ax0.legend(frameon = False)
    ax0.set_title("trial: {}".format(tr))
    ax0.set_ylabel('inferred DV')    
    ax0.set_ylim([-10,10])


    # plot DVs averaged over gammas from the two regions
    for r, reg in enumerate(regions):
            
        ax = fig_DV.add_subplot(gs[1+r, 0:2])
        for u in np.unique(task_data['gamma']):
            rtrials = np.where(np.array(task_data['gamma']) == u)[0]
            ax.plot(np.nanmean(DV[reg][rtrials, :], axis = 0), label = u)
        # ax3.legend(ncol = 2, frameon = False)
        ax.set_title('REG: {} '.format(reg))
        ax.set_ylabel('Mean DV for gammas')
        ax.set_ylim([-5,5])



    # now plot cross-correlation 
    ax = fig_DV.add_subplot(gs[3:5,0:2])
    plot_cc(DV_summary['DV_cc'], 'k', ax, 'DATA')
    ax.axvline(DV_summary['DV_cc']['peak'], c = 'k')
    plot_cc(DV_summary['DV_cc_shuff'], 'r', ax, 'shuffle')
    ax.legend(frameon = False)
    ax.axvline(0, ls = ':')
    ax.axhline(0, c = 'k',ls = ':')
    ax.set_xlim([-0.4, 0.4])
    ax.set_xlabel('Lags [s]')
    ax.set_ylabel('Decision variable \n correlation (A, B)')
    # ax.set_ylim([-0.1, 0.2])
    for r, reg in enumerate(regions):
        ax.text(-0.1 + r*0.2,  
            ax.get_ylim()[1], 
            reg + ' leads', 
            fontsize=10, 
            va='center', 
            ha='center', 
            backgroundcolor='w')


    # plot accumulation latents against each other
    ax0 = fig_DV.add_subplot(gs[1:3,2:4])
    for pR in np.unique(df_trial['pokedR']):
        idx = df_trial['pokedR'] == pR
        ax0.scatter(
            np.ravel(latents[idx,:,0]),
            np.ravel(latents[idx,:,3]), 
            marker = '.',
            alpha = 0.5,
            label = 'choice = {}'.format(pR))
    ax0.legend(frameon = False)
    ax0.set_xlabel('Accumulation latent reg A')
    ax0.set_ylabel('Accumulation latent reg B')
        
            
    # plot inferred DVs against each other
    ax0 = fig_DV.add_subplot(gs[3:5,2:4])
    for pR in np.unique(df_trial['pokedR']):
        idx = df_trial['pokedR'] == pR
        nonnan = ~np.isnan(DV['A'])[idx,:]
        plt.scatter(
            np.ravel(DV['A'][np.where(nonnan == 1)]),
            np.ravel(DV['B'][np.where(nonnan == 1)]), 
            marker = '.',
            alpha = 0.5,
            label = 'choice = {}'.format(pR))
    ax0.legend(frameon = False)
    ax0.set_xlabel('DV reg A')
    ax0.set_ylabel('DV reg B')
    ax0.set_xlim([-8,8])
    ax0.set_ylim([-8,8])
        
    sns.despine()
    # plt.tight_layout()



class PoissonClicks():
    
    def __init__(self, **params):
        
        self.dt = params.get('dt', 0.01)
        self.N_batch = params.get('N_batch', 300)
        self.T = params.get('T', [0.2,1.0])
        self.gamma_list = params.get('gamma_list', np.linspace(-3.5, 3.5, 8))
        self.history_bias = params.get('history_bias', True)
        self.total_rate = params.get('total_rate', 40)
        
        
    def generate_trial_params(self, batch, trial):
        
        prm = dict()
        prm['gamma'] = np.random.choice(self.gamma_list)
        
        if self.history_bias == True:
            prm['history_bias'] = np.random.choice([-1,1])
        else:
            prm['history_bias'] = 0
            
        prm['T'] = self.T[0] + np.random.rand()*(self.T[1] - self.T[0])
            
        return prm
    
    
    def generate_trial(self, prm, get_LR = False):
        
        # generate clicks
        rate_r = self.total_rate * np.exp(prm['gamma'])/(1+np.exp(prm['gamma'])) + 1e-16
        rate_l = self.total_rate - rate_r
        
        # click times
        click_time_r = np.cumsum(np.random.exponential(1/rate_r, 100))
        click_time_l = np.cumsum(np.random.exponential(1/rate_l, 100))
        
        # binned outputs with dimensions     
        binned_r = np.histogram(click_time_r, np.arange(0., prm['T'] + self.dt, self.dt))[0]
        binned_l = np.histogram(click_time_l, np.arange(0., prm['T'] + self.dt, self.dt))[0]
        
        prm['Δclicks'] = np.sum(binned_r) - np.sum(binned_l)
        
        if get_LR:
            return binned_l, binned_r, prm
        else:
            return binned_r - binned_l, prm
    
    
    def batch_generator(self, get_LR = False):
        
        batch = 1
        while batch > 0:
            if get_LR:
                left_click_data = []
                right_click_data = []
            else:
                click_data = []
            task_data = dict()
            
            for trial in range(self.N_batch):
                p = self.generate_trial_params(batch, trial)
                if get_LR:
                    outL, outR, p = self.generate_trial(p, get_LR)
                    left_click_data.append(outL)
                    right_click_data.append(outR)
                else:
                    out, p = self.generate_trial(p, get_LR)
                    click_data.append(out)
                
                for key in list(p):
                    if trial == 0:
                        task_data[key] = []
                    task_data[key].append(p[key])
                    
            batch += 1
            
            if get_LR:
                yield nanpad_ragged_sequence(left_click_data), nanpad_ragged_sequence(right_click_data), task_data
            else:
                yield nanpad_ragged_sequence(click_data), task_data
            
    def get_trial_batch(self, get_LR = False):
        
        return next(self.batch_generator(get_LR))
    
    
    def get_params(self):
        
        return self.__dict__
    
    
    
class NeuralSimulator():
    
    def __init__(self, **params):
        
        self.dt = params.get('dt', 0.01)
        self.N_neurons_per_region = params.get('N_neurons_per_region', 50)
        self.bound = params.get('bound', 8)
        self.leak = params.get('leak', 0.8)
        self.N_latents_per_region = params.get('N_latents_per_region', 3)
        self.interaction_type = params.get('interaction_type', 'feedforward')
        self.history_bias = params.get('history_bias', True)
        self.sigmas_init = params.get('sigmas_init', 2.)
        self.sigmas = params.get('sigmas', 40)
        self.ff_delay = params.get('ff_delay', 1) # in dt units
        
        num_neurons = 2*self.N_neurons_per_region
        num_latents = 2*self.N_latents_per_region
        
        # dynamics matrix
        self.A = np.zeros((2, num_latents, num_latents))
        
        # dynamics matrix with delay
        self.A_delay = np.zeros((2, num_latents, num_latents))
        
        # inputs matrix
        if self.history_bias == True:
            self.B = np.zeros((2, num_latents, 2))
            assert self.N_latents_per_region >= 2, 'Code is not setup to deal with fewer dims'
        else:
            self.B = np.zeros((2, num_latents))
            assert self.N_latents_per_region >= 1, 'Code is not setup to deal with fewer dims'
            
        # emissions matrix
        self.C = np.zeros((num_neurons, num_latents))
        
        # emissions bias
        self.b = 0.5 + np.random.normal(size = (num_neurons, 1))
        
        # noise multiplier for bound hitting state
        self.noise_mul = np.ones((2, num_latents))
        
        self.noise_mul[0, self.N_latents_per_region] = 0.
        self.noise_mul[1,0] = 0.
        self.noise_mul[1, self.N_latents_per_region] = 0.
        
        # map dynamics matrix based on the interaction type
        interaction_dict = {"feedforward": self.set_feedforward_params,
                            "distributed": self.set_distributed_params,
                            "recurrent": self.set_recurrent_params}
        interaction_dict[self.interaction_type]()
        
        if "dynamics_matrix" in params.keys():
            print("Dynamics matrix specified. These will be given precedence over default settings for the interaction type")
            assert np.shape(self.A) == np.shape(params['dynamics_matrix']), 'Specified dynamics matrix is incompatible'
            self.A = params.get('dynamics_matrix')
            
        if "input_matrix" in params.keys():
            print("Input matrix specified. These will be given precedence over default settings for the interaction type")    
            assert np.shape(self.B) == np.shape(params['input_matrix']), 'Specified input matrix is incompatible'
            self.B = params.get('input_matrix')
            
        if "emissions_matrix" in params.keys():
            print("Emissions matrix specified. These will be given precedence over default settings for the interaction type")
            assert np.shape(self.C) == np.shape(params['emissions_matrix']), 'Specified emissions matrix is incompatible'
            self.C = params.get('emissions_matrix')
        
        
    def set_feedforward_params(self):
        
        N_neurons = self.N_neurons_per_region
        N_latents = self.N_latents_per_region
        
        # set up autoregressive terms
        np.fill_diagonal(self.A[0], 0.97 + 0.02*np.random.rand(2*N_latents))
        
        # set up the accumulation dimension with a small trial history input
        self.A[0,0,0] = self.leak
        self.A[0,0,1] = 0.1
        
        # no accumulation in latents of other region, it only inherits - no additional noise
        self.A[0, N_latents, N_latents] = 0
        self.A_delay[0, N_latents, 0] = 1.
        self.A_delay[1, N_latents, 0] = 1.
        # self.noise_mul[0, self.N_latents_per_region] = 0.
        
        # copy the same dynamics when bound is reached, but with no accumulation anymore
        self.A[1] = self.A[0]
        self.A[1,0,0] = 1. # keep accumulation dimensions where they are
        self.A[1,0,1] = 0. # no history input
        
        # setup inputs
        self.B[0,0,0] = 1
        self.B[0, N_latents, 0] = 1
        
        if self.history_bias == True:
            self.B[0,1,1] = 1
            self.B[0,N_latents+1, 1] = 1
            
        self.C[:N_neurons, :N_latents] = 8. * np.random.normal(size = (N_neurons, N_latents))
        self.C[N_neurons:, N_latents:] = 8. * np.random.normal(size = (N_neurons, N_latents))
        
        
        
    def set_distributed_params(self):
        pass
    
    def set_recurrent_params(self):
        pass
    
    
    def simulate_trial(self, trial, clicks, task_data):
        
        st = 0
        noise_var = np.sqrt(self.dt*self.sigmas)
        stim_length = sum(~np.isnan(clicks))
        
        acc_idx = self.get_accumulator_latent_index()
        hist_idx = self.get_history_latent_index()
        AR_idx = self.get_AR_latent_index()

        for t in range(len(clicks)):
            
            if np.isnan(clicks[t]):
                self.inputs[0] = 0.
            elif clicks[t] == 0:
                self.inputs[0] = 0.
            else:
                self.inputs[0] = clicks[t] + np.random.normal()*noise_var
            
            if t == 0:
                self.z[trial,0,:] = np.sqrt(self.dt) * self.sigmas_init * np.random.randn(2*self.N_latents_per_region)
                if self.history_bias == True:
                    self.inputs[1] = task_data['history_bias'][trial] + np.random.normal()*noise_var
                self.z[trial,t,:] += np.matmul(self.B[st], self.inputs)
                
            else:
                self.inputs[1] = 0.
                self.z[trial, t, :] = np.matmul(self.A[st], self.z[trial, t-1, :]) + \
                        np.matmul(self.B[st], self.inputs) + \
                            np.random.normal(
                                loc = 0.,
                                scale = noise_var * self.noise_mul[st],
                                size = 2*self.N_latents_per_region)
                                
                if (t - self.ff_delay) >= 0:
                    self.z[trial, t, :] += np.matmul(self.A_delay[st], self.z[trial, t-self.ff_delay, :]) 
                
                # if either of the regions reach bound dynamics switch
                if np.abs(self.z[trial, t, 0]) > self.bound:
                    st = 1
                elif np.abs(self.z[trial, t, self.N_latents_per_region] > self.bound):
                    st = 1
                    
        # assign choice for this trial based on the accumulator
        task_data['choice'].append(np.sign(self.z[trial, stim_length-1, 0]) > 0)
        
            
        # normalize so that all latents are about the same value
        self.z[trial,:,acc_idx[0]] /= self.bound
        self.z[trial,:,acc_idx[1]] /= self.bound
        self.z[trial,:,hist_idx[0]] /= 1.
        self.z[trial,:,hist_idx[1]] /= 1.
        self.z[trial,:,AR_idx[0]] /= 1.
        self.z[trial,:,AR_idx[1]] /= 1.
        
        # compute spike rates and sample spikes for the neural population
        spike_rate = np.matmul(self.C, self.z[trial, :, :].T) + self.b
        self.spike_rate[trial, :, :] = self.dt * np.maximum(spike_rate,0).T

        return task_data                    
                                            
        
    def simulate_trials(self, click_data, task_data):
        
        ntrials = click_data.shape[0]
        num_tpts = click_data.shape[1]
        task_data['choice'] = []
        
        # initialize the matrix for storing evolution of latent variables, spike rates
        self.z = np.nan * np.zeros((ntrials, num_tpts, 2*self.N_latents_per_region))
        self.spike_rate = np.nan * np.zeros((ntrials, num_tpts, 2*self.N_neurons_per_region))
        self.inputs = np.zeros(self.B.shape[2])
        self.clicks = click_data
        
        for trial in range(ntrials):
            task_data = self.simulate_trial(trial, click_data[trial,:], task_data)
            
        self.sample_spikes()
        
        return self.z, self.spikes, self.spike_rate, task_data
    
    
    def sample_spikes(self, spike_rate_dt = None):
        
        if spike_rate_dt is None:
            spike_rate_dt = self.spike_rate 
            
        self.spikes = np.nan * np.zeros(np.shape(spike_rate_dt))
        nanmask = np.isnan(spike_rate_dt)
        spike_rate_dt[nanmask] = 1.
        
        spikes = np.random.poisson(lam = np.ravel(spike_rate_dt[:,:,:]))
        self.spikes = spikes.reshape(spike_rate_dt.shape[0], spike_rate_dt.shape[1], spike_rate_dt.shape[2]).astype(float)
        
        self.spikes[nanmask] = np.nan
        spike_rate_dt[nanmask] = np.nan
        self.spike_rate[nanmask] = np.nan        
    
    def get_params(self):
        
        return self.__dict__
    
    
    def get_latents_labels(self):
        
        # computing labels of latents dimension based on their number
        xlabel_dyn = []
        for i in range(2*self.N_latents_per_region):
            if i < self.N_latents_per_region:
                ap = 'A'
                val = 0
            else:
                ap = 'B'
                val = self.N_latents_per_region
                
            if (i-val) == 0:
                xlabel_dyn.append('Acc ' + ap)
            elif (i-val) == 1:
                xlabel_dyn.append('Hist ' + ap)
            else:
                xlabel_dyn.append('AR ' + ap)
                
        return xlabel_dyn
    
    
    def get_accumulator_latent_index(self):
        return [0, self.N_latents_per_region]
    
    def get_history_latent_index(self):
        return [1 , self.N_latents_per_region + 1]
    
    def get_AR_latent_index(self):
        num_latents = self.N_latents_per_region
        return flatten_list(
            [np.arange(2, num_latents), 
            np.arange(num_latents + 2, 2*num_latents)])
    

    def plot_dynamics_params(self):
        
        fig_dyn = plt.figure(constrained_layout = True, figsize = (12,8))
        gs = fig_dyn.add_gridspec(5,6)
        fig_dyn.suptitle('Simulation parameters', fontsize = 16)
        
        xlabel_dyn = self.get_latents_labels()
                
        # dynamics matrix during accumulation
        ax0 = fig_dyn.add_subplot(gs[0:2, :2])
        ax0.matshow(self.A[0], cmap = 'Blues', vmin = 0., vmax = 1.)
        ax0.set_title('Accumulation: dynamics')
        ax0.set_yticks(ticks = range(2*self.N_latents_per_region), labels = xlabel_dyn)
        ax0.set_xticks(ticks = range(2*self.N_latents_per_region), labels = xlabel_dyn, rotation = 90)
        
        # input matrix during accumulation
        ax1 = fig_dyn.add_subplot(gs[0:2, 2])
        ax1.matshow(self.B[0], cmap = 'Blues', vmin = 0., vmax = 1.)
        ax1.set_title('Input')
        ax1.set_xticks(ticks = [0.5, 2.5], labels = {'δR-δL', 'history'}, rotation = 45)
        ax1.set_ylabel('Latents')
        
        # dynamics matrix once bound has been hit
        ax00 = fig_dyn.add_subplot(gs[0:2, 3:5])
        ax00.matshow(self.A[1], cmap = 'Blues', vmin = 0., vmax = 1.)
        ax00.set_title('Bound hit - dynamics')
        ax00.set_yticks(ticks = range(2*self.N_latents_per_region), labels = xlabel_dyn)
        ax00.set_xticks(ticks = range(2*self.N_latents_per_region), labels = xlabel_dyn, rotation = 90)
        
        # input matrix once bound has been hit
        ax11 = fig_dyn.add_subplot(gs[0:2,5])
        ax11.matshow(self.B[1], cmap = 'Blues', vmin = 0., vmax = 1.)
        ax11.set_title('Input')
        ax11.set_xticks(ticks = [0.5, 2.5], labels = {'δR-δL', 'history'}, rotation = 45)
        ax11.set_xlabel('Latents')
        
        # emissions matrix
        ax2 = fig_dyn.add_subplot(gs[3,:])
        ax2.matshow(self.C.T, vmin = -2, vmax = 2, cmap = 'RdBu')
        ax2.set_xlabel('Neuron')
        ax2.set_ylabel('Latents')
        ax2.set_title('Emissions matrix')
        
        # emissions bias
        ax3 = fig_dyn.add_subplot(gs[4,:])
        ax3.bar(x = range(2*self.N_neurons_per_region), height = self.b.squeeze(), color = 'k')
        ax3.set_title('Emission bias')
        ax3.set_xlabel('Neuron')
        
        sns.despine()
        
        
        
        
    def plot_activity_portrait(self, task_data = None):
        
        if (hasattr(self, 'spikes') == False) | (task_data == None):
            raise Exception('No activity simulated, first run some trials with ns.simulate trials and provide task_data')

        fig_act = plt.figure(constrained_layout = True, figsize = (9,12))
        gs = fig_act.add_gridspec(7,3)
        fig_act.suptitle('Activity portrait', fontsize = 16)
        
        
        
        def plot_trial(ax_num):
            
            labels = self.get_latents_labels()
        
            this_trial = np.random.choice(len(task_data['gamma']))
            # plot an example trial (click trains)
            ax3 = fig_act.add_subplot(gs[ax_num,0])
            idx = np.ravel(np.argwhere(self.clicks[this_trial,:] != 0))
            ax3.scatter(idx*self.dt, self.clicks[this_trial,idx], marker='|', c= 'k')
            ax3.set_ylim([-3,3])
            ax3.set_xlabel('Time in sec')
            ax3.set_ylabel('click difference')
            ax3.set_title('Gamma: {}'.format(task_data['gamma'][this_trial]))
            
            
            # plot an example trial (latents)
            ax4 = fig_act.add_subplot(gs[ax_num,1])
            len_trial = sum(~np.isnan(self.clicks[this_trial, :]))
            palette = iter(sns.color_palette("Blues", 5, desat = 1.))
            for i in range(self.N_latents_per_region):
                color = next(palette)
                ax4.plot(np.arange(len_trial)*self.dt, self.z[this_trial, :len_trial, i], c = color, label = labels[i])
            palette = iter(sns.color_palette("Reds", 5, desat = 1.))
            for i in range(self.N_latents_per_region, 2*self.N_latents_per_region):
                color = next(palette)
                ax4.plot(np.arange(len_trial)*self.dt, self.z[this_trial, :len_trial, i], c = color, label = labels[i])
            ax4.set_xlabel('Time in sec')
            ax4.set_ylabel('latent activity [au]')
            # ax4.legend(frameon = False)
            
            # plot an example trial (firing rates)
            ax5 = fig_act.add_subplot(gs[ax_num,2])
            ax5.imshow(self.spike_rate[this_trial, :len_trial,:].T, interpolation=None)  
            ax5.set_aspect('auto')
            
            
            
        def plot_latents(ax_num, latent_labels):
            
            latent_a = np.argwhere([a == latent_labels[0] for a in self.get_latents_labels()])[0][0] 
            latent_b = np.argwhere([a == latent_labels[1] for a in self.get_latents_labels()])[0][0] 
            len_trial = self.clicks.shape[1]
            ax3 = fig_act.add_subplot(gs[ax_num, 0])
            ax4 = fig_act.add_subplot(gs[ax_num, 1])
            for u in np.unique(task_data['gamma']):
                rtrials = np.where(np.array(task_data['gamma']) == u)[0]
                ax3.plot(np.arange(len_trial)*self.dt,np.mean(self.z[rtrials, :, latent_a], axis = 0), label = u)
                ax4.plot(np.arange(len_trial)*self.dt,np.mean(self.z[rtrials, :, latent_b], axis = 0), label = u)
                # ax3.legend(ncol = 2, frameon = False)
            ax3.set_title(latent_labels[0])
            ax4.set_title(latent_labels[1])
            # ax3.legend(frameon = False)
        
        
        
        # psychometric curve
        ax0 = fig_act.add_subplot(gs[0,0])
        # fit a psychometric curve
        try:
            popt, pcov = curve_fit(
                psych_func, 
                np.array(task_data['Δclicks']).astype('float32'),
                np.array(task_data['choice']).astype('float32'),
                maxfev = 50000)
            psych = psych_func(np.linspace(-40,40, 81), *popt)
            ax0.plot(np.linspace(-40,40, 81), psych)
        except Exception as e:
            print("Error occurred during curve fitting:", e)    
        # actual data
        df = pd.DataFrame(task_data)
        Δcuts, bins = pd.cut(task_data['Δclicks'],bins = 20, retbins = True)
        bin_centers = bins[:-1] + np.diff(bins)[0]/2
        mean_values = df.groupby(Δcuts)['choice'].mean()
        ax0.scatter(bin_centers, mean_values, c = 'k', s = 1)
        ax0.set_xlabel('#R - #L')
        ax0.set_ylabel('Fraction chose right')
        ax0.set_ylim([-0.05,1.05])
        
        # distribution of firing rates
        ax1 = fig_act.add_subplot(gs[0,1])
        ax1.hist(np.nanmean(self.spike_rate[:,:,:self.N_neurons_per_region], axis = (0,1))/self.dt, label = 'A', alpha = 0.5, color = 'b')
        ax1.hist(np.nanmean(self.spike_rate[:,:,self.N_neurons_per_region:], axis = (0,1))/self.dt, label = 'B', alpha = 0.5, color = 'r')
        ax1.legend(frameon = 'false')
        ax1.set_ylabel('Mean spks/s')
        ax1.set_xlabel('Neuron Count')
        
            
        plot_trial(ax_num = 1)
        plot_trial(ax_num = 2)
        plot_trial(ax_num = 3)        
        plot_latents(ax_num = 4, latent_labels=['Acc A', 'Acc B'])
        plot_latents(ax_num = 5, latent_labels=['Hist A', 'Hist B'])
        plot_latents(ax_num = 6, latent_labels=['AR A', 'AR B'])
        
        sns.despine()



if __name__ == "__main__":
    
        
    N_neurons = np.linspace(10, 100, 10).astype(int)
    N_trials = np.linspace(100, 300, 5).astype(int)
    ff_delays = np.linspace(1, 20, 5).astype(int)
    N_repeats = 5

    # for simulations
    params = dict()
    params['dt'] = 0.001
    params['history_bias'] = True
    params['bound'] = 8.
    params['leak'] = 0.99
    params['T'] = [0.2, 1.0]



    # for decoding
    p = dict()
    p['regions'] = ['A', 'B']
    p['cols'] = SPEC.COLS
    p['fr_thresh'] = 1.0 # firing rate threshold for including neurons
    p['stim_thresh'] = 0.0 # stimulus duration threshold for including trials
    p['align_to'] = ['clicks_on']
    p['align_name'] = ['clickson_masked']
    p['pre_mask'] = [None]
    p['post_mask'] = ['clicks_off']
    p['start_time'] = [0,]
    p['end_time'] = [1000] # in ms
    p['binsize'] = 50 # in ms
    p['filter_type'] = 'gaussian'
    p['filter_w'] = 75 # in ms
    p['Cs'] = np.logspace(-7,3,200)  # cross-validation parameter
    p['nfolds'] = 10  # number of folds for cross-validation
    p['n_repeats'] = 1 # number of repeats for cross-validation


    # for DV cross correlations
    prm = dict()
    prm['regions'] = ['A', 'B']
    prm['cols'] = SPEC.COLS
    prm['align_to'] = ['clicks_on']
    prm['align_name'] = ['clickson_masked']
    prm['pre_mask'] = [None]
    prm['post_mask'] = ['clicks_off']
    prm['start_time'] = [0]
    prm['end_time'] = [800] # in ms
    prm['binsize'] = 5 # in ms
    prm['filter_type'] = 'gaussian'
    prm['filter_w'] = 50 # in ms

    decoded_variable = 'pokedR'

    FIGSAVEPATH = SPEC.FIGUREDIR + "figure2/DVsweeps/"
    DATASAVEPATH = SPEC.RESULTDIR + "figure2/DVsweeps/"
    
    
    DV_summary = dict()
    DV_summary['mdl_coefs'] = dict()
    DV_summary['corr_DVaxis_emissions'] = dict()
    DV_summary['mdl_intercept'] = dict()
    DV_summary['dec_accuracy'] = dict()
    
    # For running the loops on cluster, make a config file and
    # submit an array job
    #
    # def write_output(ff_delays, N_trials, N_neurons, output_file):
    #     with open(output_file, 'w') as file:
    #         file.write("ArrayTaskID ff_delays N_neurons N_trials\n")
    #         count = 1
    #         for i in (ff_delays):
    #             for j in (N_neurons):
    #                 for k in (N_trials):
    #                     file.write(f"{count} {i} {j} {k}\n")
    #                     count += 1
    # N_neurons = np.linspace(10, 100, 10).astype(int)
    # N_trials = np.linspace(100, 300, 5).astype(int)
    # ff_delays = np.flip(np.linspace(1, 20, 5).astype(int))
    # output_file = "output.txt"
    # write_output(ff_delays, N_trials, N_neurons, output_file)

    ff = int(sys.argv[1])
    nn = int(sys.argv[2])
    nt = int(sys.argv[3])
            
    params['N_neurons_per_region'] = nn
    params['N_batch'] = nt
    params['ff_delay'] = ff
    
    for repeat in range(N_repeats):
        
        filename = "DVsweep_num_trials_{}_num_neurons_{}_ffdelay_{}ms_repeat_{}".format(nt, nn, ff, repeat)
        
        # simulate data
        pc_task = PoissonClicks(**params)
        ns = NeuralSimulator(**params)
        click_data, task_data = pc_task.get_trial_batch()
        latents, spikes, spike_rate_dt, task_data = ns.simulate_trials(click_data, task_data)
        
        # save some plots and process data
        ns.plot_activity_portrait(task_data)
        savethisfig(FIGSAVEPATH, filename + "_activity")
        ns.plot_dynamics_params()
        savethisfig(FIGSAVEPATH, filename + "_parameters")
        df_trial, df_cell = make_dataframes(task_data, ns, spikes)
        
        # find DV axis
        target = np.array(df_trial[decoded_variable], dtype = float)
        summary = dict()
        for r, reg in enumerate(p['regions']):
            X, ntpts_per_trial = get_neural_activity(df_cell, df_trial, reg, p, 0)
            summary[reg] = run_logistic_decoding(X, target, ntpts_per_trial, p)
            DV_summary['mdl_coefs'][reg] = summary[reg]['mdl_coefs']
            DV_summary['mdl_intercept'][reg] = summary[reg]['mdl_intercept']
            DV_summary['corr_DVaxis_emissions'][reg] = np.corrcoef(
                summary[reg]['mdl_coefs'], 
                ns.C[r*nn:(r+1)*nn, r*ns.N_latents_per_region])
            DV_summary['dec_accuracy'][reg] = summary[reg]['accuracy']
                            
        # compute DVs
        X = dict()
        DV = dict()
        for reg in prm['regions']:
            X[reg], ntpts = get_neural_activity(df_cell, df_trial, reg, prm, 0)
            trial_idx = np.repeat(np.arange(nt), ntpts)
            DV[reg] = np.nan * np.zeros((nt, max(ntpts)))
            for tr in range(nt):
                this_DV = summary[reg]['mdl_coefs'] @ X[reg][:, trial_idx == tr] + summary[reg]['mdl_intercept']
                DV[reg][tr, :ntpts[tr]] = this_DV
        
        # cross correlate DVs
        DV_summary['DV_cc'] = crosscorrelate_DVs(DV, prm, shuffle = False)
        DV_summary['DV_cc_shuff'] = crosscorrelate_DVs(DV, prm, shuffle = True)
        
        # plot and save data
        plot_cross_corr_metrics(DV_summary, DV, ns, summary, df_trial, latents)
        savethisfig(FIGSAVEPATH, filename + "_DVcc")
        
        
        # Create a dictionary to hold the named dictionaries
        # all_dicts = {
        #     'decoder_p': p, 
        #     'DV_p': prm, 
        #     'ns_p': ns.get_params(), 
        #     'DV_summary': DV_summary}
        
        DV_summary['params'] = params
        DV_summary['p_decoding'] = p
        DV_summary['p_cc'] = prm
        

        # Save the dictionary containing all named dictionaries to a file
        print("\n\n\nSaving data: {}".format(DATASAVEPATH + filename + ".npy"))
        np.save(DATASAVEPATH + filename + ".npy", DV_summary)

        




                    

        

        
            
                