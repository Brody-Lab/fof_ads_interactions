import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import copy
import seaborn as sns
import warnings
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

from helpers.phys_helpers import *

# TODOS:
# [ ] pass figure/ axis handles for plotting


def adjust_window(window, binsize):
    '''Helper function which adjusts the window so as to be compatible with binsize
    Args:
        window (list of two elements): time interval, 
        elements are beginning and ending of the interval
        binsize (float): binsize
    Returns:
        window: again a list with two elements, 
        adjusted to have equal binsizes across the window
    Edits:
        Sept 14, 2021 DG: to not change the starting point of window
        but instead truncate the end to be an even multiple of binsize
    '''
#     window = [int(np.floor(window[0] / binsize) * binsize),
#               int(np.ceil(window[1] / binsize) * binsize)]

    if np.mod(window[1] - window[0], binsize) != 0:
        window[1] = window[1] - np.mod(window[1] - window[0], binsize)

    return window


def adjust_window_for_convolution(window, spikefilter, binsize):
    '''Helper function which pads the window for convolution
    Args:
        window (list of two elements): original time interval, 
        elements are beginning and ending of the interval
        spikefilter (list): 1D filter
        binsize (float): interval over which spike filter has been discretized
    Returns:
        padded window
    '''
    return [(window[0] - ((len(spikefilter)-1)/2)*binsize),
            (window[1] + ((len(spikefilter)-1)/2)*binsize)]


def make_raster(spiketimes, df, align_to, split_by=None,
                window=[-500, 1000], binsize=10,
                pre_mask=None, post_mask=None,
                plot=True, cmap='binary'):
    '''Makes the raster and optionally plots it.
    Args:
        spiketimes (1D numpy array): spiketimes (in seconds) typically for 1 cell
        df (DataFrame): dataframe containing the (trial) data, `align_to` and `split_by` are
            referenced from this dataframe 
        align_to (str): Column name of dataFrame `df` for aligning the raster. 
            Values are assumed to be in s 
        split_by (str): Column name in the dataframe `df`, based on unique values of trials
            indices will be grouped. 
            If plotting, each split_by group is plotted as a separate subplot
        window (list of two elements): Time interval (in ms) for the raster centered around `align_to`
        binsize (int): binsize (in ms)
        pre_mask (str): Column name of dataFrame `df` for pre masking the raster i.e. spikes which happen 
            before this event will be replaced by nans. Values are assumed to be in s 
        post_mask (str): Column name of dataFrame `df` for post masking the raster i.e. spikes which happen 
            after this event will be replaced by nans. Values are assumed to be in s 
        plot (bool): plot if `True` calls `plot_raster`
        cmap (str): Matplotib colormap name for plotting the raster
    Returns:
        raster_summary (dict): with keys `align_to`, `split_by`, `window`, `binsize`, `data`
            and if `plot` was `True`, `fig` and `axs` handles
            `raster_summary['data']` is a dictionary where each key is a raster for each 
            unique entry of `df[split_by]`
    '''

    # formatting data
    spiketimes = np.squeeze(np.sort(spiketimes))

    # group trials
    trials = split_trials(df, split_by)
    conditions = np.sort(list(trials.keys()))

    # make bins
    window = adjust_window(window, binsize)
    edges = np.arange(
        window[0],
        window[1] + binsize,
        binsize)*0.001  # convert to s

    # make raster for each split_by condition
    rasters = dict()
    masks = dict()
    for cond in conditions:

        # for each trial infer aligning and masking timepoints
        align_times = df[align_to][trials[cond]]
        if pre_mask is None:
            pre_mask_times = [min(align_times-1000)]*len(align_times)
        else:
            pre_mask_times = df[pre_mask][trials[cond]]
        if post_mask is None:
            post_mask_times = [max(align_times+1000)]*len(align_times)
        else:
            post_mask_times = df[post_mask][trials[cond]]

        # compute raster for the condition
        raster = []
        mask = []
        for t, m_pre, m_post in zip(align_times, pre_mask_times, post_mask_times):
            idx = np.squeeze(np.searchsorted
                             (spiketimes, [t + edges[0], t + edges[-1]]))
            counts, binedges = np.histogram(
                spiketimes[idx[0]:idx[1]], t + edges)
            # converting to float to support nan conversion
            raster.append([float(i) for i in counts])

            # compute the mask
            mask_vec = [1] * len(counts)
            idx_mask_pre = np.searchsorted(binedges, m_pre)
            idx_mask_post = np.searchsorted(binedges, m_post)
            mask_vec[:idx_mask_pre] = [np.nan] * idx_mask_pre
            mask_vec[idx_mask_post:] = [np.nan] * \
                (len(mask_vec) - idx_mask_post)
            mask.append(mask_vec)

        rasters[cond] = np.array(raster)
        masks[cond] = np.array(mask)

    raster_summary = {
        'align_to': align_to,
        'split_by': split_by,
        'window': window,
        'binsize': binsize,
        'pre_mask': pre_mask,
        'post_mask': post_mask,
        'data': rasters,
        'mask': masks
    }

    if plot is True:
        fig, axs = plot_raster(copy.deepcopy(raster_summary), cmap=cmap)
        raster_summary['fig'] = fig
        raster_summary['axs'] = axs

    return raster_summary


def plot_raster(data_dict, conditions=None, cmap='binary'):
    '''Plot a single raster, appropriately titles the alignment and grouping
    Args:
        data_dict (dict): dictionary with rasters, output of `make_raster` function
        conditions (list): keys from data_dict['data'] which will be plotted as rasters -
            each condition is plotted as a different subplot. If no conditions are specified 
            then all the keys are plotted
        cmap (str): Matplotib colormap name for plotting the raster
    Returns:
        figure and axis handles
    '''

    # unpacking
    align_to = data_dict['align_to']
    split_by = data_dict['split_by']
    window = [w*0.001 for w in data_dict['window']]  # convert to s
    binsize = data_dict['binsize'] * 0.001  # convert to s
    if conditions is None:
        conditions = list(data_dict['data'].keys())

    # this turned out to be tricky for some reason so putting ticks very slectively
    xticks = [-0.5, (-window[0]) / binsize - 0.5,
              (window[1] - window[0]) / binsize - 0.5]
    xticks_label = [window[0], 0, window[1]]

    # function which does the actuall plotting
    def _plot_raster(cond, ax):
        data_masked = apply_mask(
            data_dict['data'][cond], data_dict['mask'][cond])
        ax.imshow(data_masked, aspect='auto', cmap=plt.get_cmap(cmap))
        ax.axvline((-window[0]) / binsize - 0.5, color='k', ls='--', lw=0.5)
        ax.set_ylabel('Trials')
        ax.set_xlabel('Time [in s]')
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(i) for i in xticks_label])
        if split_by is not None:
            ax.set_title(split_by + " = " + str(cond), fontsize=10)
        sns.despine()

    # axs is not iterable if there is only 1 subplot, so this hula hoop
    if hasattr(conditions, "__len__") is False:
        fig, axs = plt.subplots(1, 1, figsize=(4, 3))
        _plot_raster(conditions, axs)
    else:
        plotnum = int(np.ceil(len(conditions)/2))
        fig, axs = plt.subplots(plotnum, 2, figsize=(8, 3*plotnum))
        for cond, axnow in zip(conditions, axs.flat):
            _plot_raster(cond, axnow)

    fig.suptitle("Aligned to : " + align_to)
    plt.tight_layout()

    return fig, axs


def make_psth(spiketimes, df, align_to=None, split_by=None,
              window=[-500, 1000], binsize=25,
              filter_type=None, filter_w=0.1,
              pre_mask=None, post_mask=None,
              plot=True):
    '''Makes the psth and optionally plots it.
    Args:
        spiketimes (1D numpy array): spiketimes (in seconds) typically for 1 cell
        df (DataFrame): dataframe containing the (trial) data, `align_to` and `split_by` are
            referenced from this dataframe 
        align_to (str): Column name of dataFrame `df` for aligning the raster. 
            Values are assumed to be in s 
        split_by (str): Column name in the dataframe `df`, based on unique values of trials
            indices will be grouped. 
        window (list of two elements): Time interval (in s) for the raster centered around `align_to`
        binsize (float): binsize (in ms). default 25ms
        filter_type (str): if filtering, which filter to use - `gaussian` or `half_gaussian`
            By default data is divided into discrete bins and not filtered
        filter_w (float): filter size argument (~in s), sets the std and therefore length of
             gaussian and half_gaussian. std is filter_w/binsize, filter extends from -5 to 5 std 
             (default: 0.1s)
        plot (bool): plot if `True` calls `plot_psth`
    Returns:
        psth_summary (dict): with keys `align_to`, `split_by`, `window`, `binsize`, `data`, 
            `filter_type`, `filter_w` and if `plot` was `True`, `fig` and `axs` handles
            `psth_summary['data']` is a dictionary where each key is a psth for each 
            unique entry of `df[split_by]`. Contains mean, std and sem as further keys
    '''

    # formatting data
    spiketimes = np.squeeze(np.sort(spiketimes))
    window = adjust_window(window, binsize)

    # get filter if we are filtering
    if filter_type is None:
        normalizer = binsize*0.001  # in s
        window_adj = window
    else:
        normalizer = 1
        filter_options = {'gaussian': gaussian,
                          'half_gaussian': half_gaussian,
                          'box_car': box_car}
        if filter_type not in filter_options:
            raise Exception('Filter type not implemented!')
        spikefilter, filter_w, binsize = filter_options[filter_type](
            filter_w, binsize)
        window_adj = adjust_window_for_convolution(
            window, spikefilter, binsize)

    raster_summary = make_raster(
        spiketimes,
        df,
        plot=False,
        align_to=align_to,
        split_by=split_by,
        window=window_adj,
        binsize=binsize)

    # compute it again with normal window for masking
    mask_summary = make_raster(
        spiketimes,
        df,
        plot=False,
        align_to=align_to,
        split_by=split_by,
        window=window,
        binsize=binsize,
        pre_mask=pre_mask,
        post_mask=post_mask)

    # make psth for each split_by condition
    psths = dict()
    trial_fr = dict()
    conditions = np.sort(list(raster_summary['data'].keys()))
    for cond in conditions:
        psths[cond] = dict()
        if filter_type is None:
            raster = raster_summary['data'][cond]
        else:
            raster = np.array([np.convolve(raster_summary['data'][cond][i],
                                           spikefilter,
                                           mode='valid')
                               for i in range(np.shape(raster_summary['data'][cond])[0])])
        raster = apply_mask(raster, mask_summary['mask'][cond])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            psths[cond]['mean'] = np.nanmean(raster, axis=0) / normalizer
            psths[cond]['std'] = (np.sqrt(np.nanvar(raster, axis=0))) / normalizer
            num_active_trials = [sum(~np.isnan(raster[:, t])) for t in range(np.shape(raster)[1])]
            psths[cond]['sem'] = psths[cond]['std'] / \
                np.sqrt(num_active_trials)

        # turn rasters into firing rates, divide spike count by binsize
        if filter_type is None:
            trial_fr[cond] = raster*1000/binsize
        else:
            trial_fr[cond] = raster

    psth_summary = {
        'align_to': align_to,
        'split_by': split_by,
        'window': window,
        'binsize': binsize,
        'pre_mask': pre_mask,
        'post_mask': post_mask,
        'data': psths,
        'trial_fr': trial_fr,
        'mask': mask_summary['mask'],
        'filter_type': filter_type,
        'filter_w': filter_w
    }

    if plot is True:
        fig, axs = plot_psth(copy.deepcopy(psth_summary), conditions=None)
        psth_summary['fig'] = fig
        psth_summary['axs'] = axs

    return psth_summary


def plot_psth(data_dict, conditions=None, ax=None, colors=None, show_legend=True):
    '''Plot PSTH, appropriately titles the alignment and grouping
    Args:
        data_dict (dict): dictionary with psths, output of `make_psth` function
        conditions (list): keys from data_dict['data'] which will be plotted as psths
            If no conditions are specified then all the keys are plotted on the same plot
        ax (matplotlib.axes.Axes, optional): axis to plot on. If None, creates new figure
        colors (list, dict, or iterator, optional): colors for each condition. Can be:
            - list of colors (same order as conditions)
            - dict mapping condition names to colors
            - iterator of colors (e.g., from seaborn palette)
            - None (uses default Spectral colormap)
        show_legend (bool): whether to show legend. Default True
    Returns:
        figure and axis handles
    '''
    # unpacking
    align_to = data_dict['align_to']
    split_by = data_dict['split_by']
    window = [w*0.001 for w in data_dict['window']]  # convert to s
    binsize = data_dict['binsize'] * 0.001  # convert to s
    if conditions is None:
        conditions = list(data_dict['data'].keys())

    x_time = np.arange(window[0], window[1], binsize) + (binsize / 2)
    if split_by is None:
        titlestr = "Aligned to : " + align_to
    else:
        titlestr = "Aligned to : " + align_to + " | Plotting : " + split_by

    # Use provided axis or create new figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Handle colors
    if colors is None:
        # Use Spectral colormap as default
        spectral_colors = plt.cm.Spectral(np.linspace(0, 1, 8))
        color_list = [spectral_colors[i % 8] for i in range(len(conditions))]
    elif isinstance(colors, dict):
        color_list = [colors.get(cond, None) for cond in conditions]
    elif isinstance(colors, list):
        color_list = colors
    elif hasattr(colors, '__iter__') and not isinstance(colors, (str, dict)):
        # Handle iterators (like seaborn palettes)
        try:
            color_list = [next(colors) for _ in range(len(conditions))]
        except StopIteration:
            # If iterator is exhausted, fall back to default
            spectral_colors = plt.cm.Spectral(np.linspace(0, 1, 8))
            color_list = [spectral_colors[i % 8] for i in range(len(conditions))]
    else:
        color_list = None

    # plot for each condition
    for i, cond in enumerate(conditions):
        psth = data_dict['data'][cond]
        
        # Get color for this condition
        if color_list is not None:
            color = color_list[i]
        else:
            color = None
            
        ax.fill_between(x_time[:len(psth['mean'])], psth['mean'] - psth['sem'],
                        psth['mean'] + psth['sem'], alpha=0.25, color=color)
        ax.plot(x_time[:len(psth['mean'])], psth['mean'], linewidth=1, label=str(cond), color=color)

    # ax.axvline(0, linestyle='--', color='k')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Firing rate [spikes/s] $\pm$ SEM')
    if len(conditions) > 1 and show_legend:
        ax.legend(
            title=split_by,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            frameon=False)
    ax.set_title(titlestr, fontsize=10)
    sns.despine(ax=ax)

    return fig, ax

def make_population_psth(df_cell, df_trial, align_to, split_by=None,
                         window=[-500, 1000], binsize=25,
                         pre_mask=None, post_mask=None,
                         filter_type=None, filter_w=None):
    '''Makes the psth for all cells in a population, options to align and split into groups
    Args:
        df_cell (Dataframe): dataframe containing neurons for whom psth will be computes
            Must include spike times in s as a list in column `spiketime_s`
        df_trial (DataFrame): dataframe containing the trial data, `align_to` and `split_by` are
            referenced from this dataframe 
        align_to (str): Column name of dataFrame `df` for aligning the raster. 
            Values are assumed to be in s 
        split_by (str): Column name in the dataframe `df`, based on unique values of trials
            indices will be grouped. Also in s
        window (list of two elements): Time interval (in s) for the raster centered around `align_to`
        binsize (float): binsize (in ms). default 25ms
        filter_type (str): if filtering, which filter to use - `gaussian` or `half_gaussian`
            By default data is divided into discrete bins and not filtered
        filter_w (float): filter size argument (~in s), sets the std and therefore length of
             gaussian and half_gaussian. std is filter_w/binsize, filter extends from -5 to 5 std 
             (default: 0.1s)
    Returns:
        psth_summary (dict): with keys `align_to`, `split_by`, `window`, `binsize`, `data`, 
            `filter_type`, `filter_w` and if `plot` was `True`, `fig` and `axs` handles
            `psth_summary['data']` is a dictionary where each key is a psth for each 
            unique entry of `df[split_by]`. Contains mean (n_neurons x n_time_bins), 
            std and sem as further keys  
    '''

    # initialize population psth
    pop_psth = dict()
    trials = split_trials(df_trial, split_by)
    for cond in trials:
        pop_psth[cond] = {name: [] for name in ['mean', 'std', 'sem']}

    # compute psth for each cell and append
    for c in range(len(df_cell)):
        cell_psth = make_psth(df_cell['spiketime_s'][c],
                              df_trial,
                              align_to=align_to,
                              split_by=split_by,
                              pre_mask=pre_mask,
                              post_mask=post_mask,
                              window=window,
                              binsize=binsize,
                              filter_type=filter_type,
                              filter_w=filter_w,
                              plot=False)
        for cond in list(cell_psth['data'].keys()):
            pop_psth[cond]['mean'].append(cell_psth['data'][cond]['mean'])
            pop_psth[cond]['std'].append(cell_psth['data'][cond]['std'])
            pop_psth[cond]['sem'].append(cell_psth['data'][cond]['sem'])

    for cond in list(pop_psth.keys()):
        pop_psth[cond]['mean'] = np.stack(pop_psth[cond]['mean'])
        pop_psth[cond]['std'] = np.stack(pop_psth[cond]['std'])
        pop_psth[cond]['sem'] = np.stack(pop_psth[cond]['sem'])

    pop_psth_summary = {
        'align_to': align_to,
        'split_by': split_by,
        'window': window,
        'binsize': binsize,
        'pre_mask': pre_mask,
        'post_mask': post_mask,
        'data': pop_psth,
        'filter_type': filter_type,
        'filter_w': filter_w
    }

    return pop_psth_summary


def plot_pop_sequence(data_dict, conditions=None, cmap_name='Blues',
                      sort_by="latency", sort_key=None, order="increasing",
                      normalize="neuron", normalize_across_cond=True):
    '''Plots the sequence plot for a neural population. Options to normalize and sort
    Args:
        data_dict (dict): dictionary with psths, output of `make_population_psth` function
        conditions (list): keys from data_dict['data'] which will be plotted
            If no conditions are specified then all the keys are plotted as subplots
        cmap_name (str): Matplotib colormap name for plotting the raster
        sort_by (str): while plotting sort neurons based on psth - can be "rate" or "latency"
            If None, neurons are not sorted
        sort_key (str): condition based on which sorting will be performed. 
            Must correspond to keys in data_dict['data']. If value is None but sort_by
            is not None, first key from data_dict['data'] is used for sorting. 
            For instance, if psths have been split by pokedR and you want to sort based 
            on leftward choices or pokedR = 0, sorting key will be 0
        order (str): sorting order: "increasing" or "decreasing". Increasing by default
        normalize (str): can be None, "neuron" or "population". If "neuron" each neuron's psth 
            is normalized to its own peak value. If "population" then each neuron's psth is 
            normalized to maximum population firing rate
        normalize_across_cond (bool): if normalization should be performed across all conditions,
            or separately for each condition
    Returns:
        fig and axs handles
        summary_dict: summary of all operations performed
    '''
    # unpacking
    align_to = data_dict['align_to']
    split_by = data_dict['split_by']
    window = [w*0.001 for w in data_dict['window']]  # convert to s
    binsize = data_dict['binsize'] * 0.001  # convert to s

    xticks = [-0.5, (-window[0]) / binsize - 0.5,
              (window[1] - window[0]) / binsize - 0.5]
    xticks_label = [window[0], 0, window[1]]
    cmap = plt.get_cmap(cmap_name)

    # normalize data
    norm_data = normalize_data(data_dict['data'],
                               normalize=normalize,
                               across_cond=normalize_across_cond)

    if conditions is None:
        conditions = np.sort(list(norm_data.keys()))

    # actual plotting
    def _plot_pop_sequence(ax, cond):
        sorted_data = norm_data[cond][sort_idx, :]
        cax = ax.imshow(sorted_data, aspect='auto', cmap=cmap, norm=normalizer)
        ax.axvline((-window[0]) / binsize - 0.5, color='k', ls='--', lw=0.5)
        ax.set_ylabel('Neurons')
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(i) for i in xticks_label])
        ax.set_frame_on(False)
        if split_by is not None:
            ax.set_title(split_by + " = " + str(cond), fontsize=10)
        sns.despine()
        return cax

    # if plotting only 1 condition
    if hasattr(conditions, "__len__") is False:
        sort_idx = get_sort_indices(norm_data[sort_key],
                                    sort_by=sort_by,
                                    order=order)
        cmax = np.amax(norm_data[conditions])
        cmin = np.amin(norm_data[conditions])
        normalizer = mpl.colors.Normalize(cmin, cmax)

        plotsize = np.shape(norm_data[conditions])
        fig, axs = plt.subplots(1, 1,
                                figsize=(2+(0.01*plotsize[0]), 3+(0.01*plotsize[1])))
        cax = _plot_pop_sequence(axs, conditions)
        cb = plt.colorbar(cax, ax=axs)

    else:

        # if plotting multiple conditions
        if sort_key is None:
            sort_key = list(norm_data.keys())[0]
        sort_idx = get_sort_indices(norm_data[sort_key],
                                    sort_by=sort_by,
                                    order=order)

        plotnum = int(np.ceil(len(conditions)/2))
        psize = np.shape(norm_data[conditions[0]])
        fig, axs = plt.subplots(plotnum, 2,
                                figsize=(4+(0.01*psize[0]), plotnum*(3+(0.01*psize[1]))))
        plt.subplots_adjust(top=0.85, wspace=0.4)

        # setting a common color map across different subplots
        cmax = max([np.amax(norm_data[cond]) for cond in conditions])
        cmin = min([np.amin(norm_data[cond]) for cond in conditions])
        normalizer = mpl.colors.Normalize(cmin, cmax)
        im = mpl.cm.ScalarMappable(norm=normalizer, cmap=cmap)

        for condnow, axnow in zip(conditions, axs.flat):
            _plot_pop_sequence(axnow, condnow)

        cb = fig.colorbar(im, ax=axs.ravel().tolist())
        axs.flat[-2].set_xlabel('Time [in s]')
        axs.flat[-1].set_xlabel('Time [in s]')

    cb.outline.set_visible(False)
    fig.suptitle("Aligned to : " + align_to, fontsize=9, y=0.98)

    plot_summary = {
        'sort_by': sort_by,
        'sort_key': sort_key,
        'order': order,
        'normalize': normalize,
        'normalize_across_cond': normalize_across_cond
    }

    return fig, axs, plot_summary


def plot_pop_psth(data_dict, conditions=None,
                  normalize="neuron", normalize_across_cond=False):
    '''Plots the average response of a neural population. Options to normalize responses
    Args:
        data_dict (dict): dictionary with psths, output of `make_population_psth` function
        conditions (list): keys from data_dict['data'] which will be plotted
            If no conditions are specified then all the keys are plotted 
        normalize (str): can be None, "neuron" or "population". If "neuron" each neuron's psth 
            is normalized to its own peak value. If "population" then each neuron's psth is 
            normalized to maximum population firing rate
        normalize_across_cond (bool): if normalization should be performed across all conditions,
            or separately for each condition
    Returns:
        fig and axs handles
    '''
    # unpacking
    align_to = data_dict['align_to']
    split_by = data_dict['split_by']
    window = [w*0.001 for w in data_dict['window']]  # convert to s
    binsize = data_dict['binsize'] * 0.001   # convert to s

    if conditions is None:
        conditions = list(data_dict['data'].keys())

    # normalize data
    norm_data = normalize_data(data_dict['data'],
                               normalize=normalize,
                               across_cond=normalize_across_cond)

    x_time = np.arange(window[0], window[1], binsize) + (binsize / 2)
    if split_by is None:
        titlestr = "Aligned to : " + align_to
    else:
        titlestr = "Aligned to : " + align_to + " | Plotting : " + split_by

    plt.figure()
    for cond in conditions:
        psth_mean = np.mean(norm_data[cond], axis=0)
        psth_std = (np.sqrt(np.var(norm_data[cond], axis=0)))
        psth_sem = psth_std / np.sqrt(float(len(norm_data[cond])))
        plt.fill_between(x_time, psth_mean - psth_sem,
                         psth_mean + psth_sem, alpha=0.25)
        plt.plot(x_time, psth_mean, linewidth=1.5, label=str(cond))

    plt.axvline(0, linestyle='--', color='k')
    plt.xlabel('Time [s]')
    plt.ylabel('Firing rate [spikes/s] $\pm$ SEM')
    if len(conditions) > 1:
        plt.legend(
            title=split_by,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            frameon=False)
    plt.title(titlestr, fontsize=10)
    sns.despine()

    return plt.gcf(), plt.gca()


# def get_sort_indices(data, sort_by="rate", order="decreasing"):
#     '''Sorts neuron based on firing rate or latency. 
#     Args:
#         data (2D nparray): Array with shape (n_neurons, n_timebins). 
#         sort_by (str): If "rate", sort by firing rate. If "latency", 
#             sort by peak latency. if None, data is returned as-is.
#         order (str): Can be "increasing" or "descreasing" 
#     Returns:
#         sorting indices 
#     '''

#     def rate(x): return np.sum(x, axis=1).argsort()
#     def latency(x): return np.argmax(x, axis=1).argsort()
#     def nosort(x): return np.arange(x.shape[0])
#     def inc(y): return y
#     def dec(y): return y[::-1]

#     sort_options = {'rate': rate,
#                     'latency': latency,
#                     None: nosort}
#     order_options = {'increasing': inc,
#                      'decreasing': dec,
#                      None: inc}
#     if sort_by not in sort_options:
#         raise Exception('Sort type not supported!')
#     if order not in order_options:
#         raise Exception('Order type not supported!')

#     return order_options[order](sort_options[sort_by](data))



def get_sort_indices(data, sort_by="rate", order="decreasing", smoothing_sigma=1, peak_prominence=None):
    """
    Returns indices to sort neurons by total firing rate or peak latency.

    Args:
        data (np.ndarray): 2D array (n_neurons, n_timebins).
        sort_by (str): One of {'rate', 'latency', None}.
        order (str): One of {'increasing', 'decreasing', None}.
        smoothing_sigma (float): Std for Gaussian smoothing of each neuron's activity (used for latency).
        peak_prominence (float or None): If set, used to filter peaks in latency detection.

    Returns:
        np.ndarray: Sorted indices of neurons.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Input `data` must be a 2D NumPy array.")

    sort_by = sort_by.lower() if sort_by else None
    order = order.lower() if order else "increasing"

    valid_sort = {"rate", "latency", None}
    valid_order = {"increasing", "decreasing"}

    if sort_by not in valid_sort:
        raise ValueError(f"`sort_by` must be one of {valid_sort}")
    if order not in valid_order:
        raise ValueError(f"`order` must be one of {valid_order}")

    def sort_by_rate(x):
        return np.argsort(np.sum(x, axis=1))

    def sort_by_latency(x):
        latencies = []
        for i in range(x.shape[0]):
            trace = gaussian_filter1d(x[i], sigma=smoothing_sigma)
            peaks, properties = find_peaks(trace, prominence=peak_prominence)

            if len(peaks) > 0:
                latency = peaks[0]  # first significant peak
            else:
                latency = np.argmax(trace)  # fallback
            latencies.append(latency)
        return np.argsort(latencies)

    def no_sort(x):
        return np.arange(x.shape[0])

    sort_funcs = {
        "rate": sort_by_rate,
        "latency": sort_by_latency,
        None: no_sort
    }

    sort_func = sort_funcs[sort_by]
    sort_indices = sort_func(data)

    if order == "decreasing":
        sort_indices = sort_indices[::-1]

    return sort_indices



def normalize_data(data, normalize, across_cond=True):
    '''Normalizes all PSTH data
    Args:
        data (dict): corresponds to the data key returned by make_population_psth. 
            Has psths (n_neurons, n_timebins) for each condition: data[cond]['mean']
        normalize (str): can be None, "neuron" or "population". If "neuron" each neuron's psth 
            is normalized to its own peak value. If "population" then each neuron's psth is 
            normalized to maximum population firing rate
        across_cond (bool): if normalization should be performed across all conditions,
            or separately for each condition
    Returns:
        dict: normalized psths for each condition, each condition is a key
    '''

    conditions = list(data.keys())

    if normalize is None:
        norm_data = {cond: data[cond]['mean'] for cond in conditions}
        return norm_data
    elif normalize not in ['population', 'neuron']:
        raise ValueError('Invalid normalization option')

    n_neurons, ntrials = np.shape(data[conditions[0]]['mean'])
    norm_data = {cond: np.empty((n_neurons, ntrials)) for cond in conditions}

    def _zero_divide(vec, divisor):
        if divisor == 0:
            return vec * 0
        else:
            return vec / divisor

    def _normalize_condition(mat, condnow):
        if normalize == "neuron":
            norm_temp = [_zero_divide(mat[i, :], np.amax(mat[i, :]))
                         for i in range(n_neurons)]
            return np.array(norm_temp)
        elif normalize == "population":
            norm_denom = np.ones([mat.shape[0], 1]) * \
                np.amax(np.amax(mat, axis=1))
            return mat / norm_denom

    if across_cond is True and hasattr(conditions, "__len__") is True:
        if normalize == "neuron":
            for i in range(n_neurons):
                norm_denom = max([np.amax(data[cond]['mean'][i, :])
                                 for cond in conditions])
                for cond in conditions:
                    norm_data[cond][i, :] = _zero_divide(
                        data[cond]['mean'][i, :], norm_denom)
        elif normalize == "population":
            norm_denom = max([np.amax(data[cond]['mean'][i, :])
                             for cond in conditions for i in range(n_neurons)])
            norm_data = {cond: data[cond]['mean'] /
                         norm_denom for cond in conditions}

    elif across_cond is False and hasattr(conditions, "__len__") is True:
        norm_data = {cond: _normalize_condition(
            data[cond]['mean'], cond) for cond in conditions}

    else:
        norm_data[conditions] = _normalize_condition(
            data[conditions]['mean'], conditions)

    return norm_data


def get_neural_activity(df_cell, df_trial, reg, p, align_id, plotting=False):
    """_summary_

    Args:
        df_cell (_type_): _description_
        df_trial (_type_): _description_
        reg (_type_): _description_
        p (_type_): _description_
        align_id (_type_): _description_
        plotting (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    align_to = p['align_to'][align_id]
    window = [p['start_time'][align_id], p['end_time'][align_id]]
    pre_mask = p['pre_mask'][align_id]
    post_mask = p['post_mask'][align_id]

    mask = 'region == @reg'
    print("\n\nCell count: {}".format(len(df_cell.query(mask))))

    X = []
    for i in range(len(df_cell.query(mask))):
        fr = make_psth(
            df_cell.query(mask).iloc[i]['spiketime_s'],
            df_trial,
            align_to=align_to,
            window=window,
            binsize=p['binsize'],
            pre_mask=pre_mask,
            post_mask=post_mask,
            filter_w=p['filter_w'],
            filter_type=p['filter_type'],
            plot=False)

        m = np.nanmean(np.ravel(fr['trial_fr'][0]))
        s = np.nanstd(np.ravel(fr['trial_fr'][0]))
        stnd_fr = (fr['trial_fr'][0] - m) / s
        # stnd_fr = stnd_fr - np.nanmean(stnd_fr, axis = 0) # subtract the mean across trials
        X.append(stnd_fr)

    X = np.array(X)
    ntpts_per_trial = np.array([sum(~np.isnan(X[0][i, :]))
                               for i in range(len(df_trial))])
    a, b, c = np.shape(X)
    X = np.reshape(X, (a, b*c))
    X = X[:, ~np.all(np.isnan(X), axis=0)]

    return X, ntpts_per_trial
