import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def get_sortCells_for_rat(ratname, DATADIR):
    '''fetched filename of Cells files for a given rat.
    '''
    datadir = DATADIR + ratname
    files = sorted([fn for fn in os.listdir(datadir) if fn.startswith((ratname))])

    return files, datadir


def savethisfig(path, name, makedir_forpath = True):
    '''saves the current figure as pdf with the given name. 
    Prints the location at whcih figure was saved.
    '''
    isExist = os.path.exists(path)
    if not isExist & makedir_forpath:
        # Create a new directory because it does not exist
        os.makedirs(path)
    elif (isExist == False) & (makedir_forpath == False):
        raise Exception("Path specified for saving figures does not exist!")
    
    filename = path + name + ".pdf"
    plt.savefig(filename, 
            dpi=300, 
            format='pdf', 
            bbox_inches='tight',
            transparent = "true")
    print("YESS QUEEN: file saved! \nJust look over here: " + filename )



def apply_mask(base_mat, mask_mat):
    '''takes two matrices, which have to be of the same size,
    finds the indices of nan values in the second matrix, 
    and sets those values to nan in the first matrix
    '''
    if base_mat.shape == mask_mat.shape: 
        base_mat[np.isnan(mask_mat)] = np.nan    
        return base_mat
    else:
        raise Exception('mask is not the same size as base array!')
        
    


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
    
    
    
def gaussian(width = 100, binsize = 25):
    '''Makes a normalized gaussian filter for spike smoothing with 
    std width/binsize, filter extends to 5*std
    Copied from make_rate_functions.m by Tim Hanks
    Args:
        width (float): in ms
        binsize (float): in ms
    Returns:
        gaussian filter
        width: bookkeeping
        binsize: bookkeeping
    '''
    from scipy.stats import norm
    
    if isinstance(binsize, int) is not True:
        raise Exception('make sure inputs are in ms')

    dx = np.ceil(4*width/binsize)
    kernel = norm.pdf(np.linspace(-dx, dx, int(dx)*2 +1), 
                         loc=0, scale = width/binsize)
    kernel = kernel * 1e3 / sum(kernel) / binsize
    return kernel, width, binsize



def half_gaussian(width = 100, binsize = 25):
    '''Makes a normalized half gaussian filter for spike smoothing with 
    std width/binsize, filter extends to 5*std
    Copied from make_rate_functions.m by Tim Hanks
    Args:
        width (float): in ms
        binsize (float): in ms
    Returns:
        gaussian filter
        width: bookkeeping
        binsize: bookkeeping
    '''
    from scipy.stats import norm
    
    if isinstance(binsize, int) is not True:
        raise Exception('make sure inputs are in ms')

    dx = np.ceil(5*width/binsize)
    np.linspace(-dx, dx, int(dx)*2 +1)
    kernel = norm.pdf(np.linspace(-dx, dx, int(dx)*2 +1), 
                         loc=0, scale = width/binsize)
    kernel[:int(dx)+1] = 0    
    kernel = kernel * 1e3 / sum(kernel) / binsize
    return kernel, width, binsize


def box_car(width = 100, binsize = 25):
    
    import scipy.signal as signal
    
    if isinstance(binsize, int) is not True:
        raise Exception('make sure inputs are in ms')
    
    kernel = signal.boxcar(int(width/binsize)) 
    kernel = kernel * 1e3 / sum(kernel) /binsize
    return kernel, width, binsize


    
def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

    
def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")
        
        
def datetime():
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%H-%M_%d-%m-%Y")




def equalize_neurons_across_regions(df_cell, regions):
    
    if len(regions) != 2:
        raise Exception('i am meant to handle two regions, gimme name of *two* regions')
        
    np.random.seed(10)

    num_cells = []
    for reg in regions:
        num_cells.append(len(np.where(df_cell['region'].str.match(reg))[0]))
    reg_more_neurons = regions[num_cells.index(np.max(num_cells))]
    idx_neurons = np.where(df_cell['region'].str.match(reg_more_neurons))[0]

    prob, binedges = np.histogram(df_cell.loc[idx_neurons, 'side_pref_p'], 20)
    binedges[-1] += 10**-6
    bin_id = np.digitize(df_cell.loc[idx_neurons, 'side_pref_p'], 
                        binedges.astype('float'), right = False) - 1
    prob = prob/len(idx_neurons)

    sub_idx = []
    while len(sub_idx) < np.min(num_cells):
        sample = np.random.randint(0,np.max(num_cells))
        if idx_neurons[sample] not in sub_idx:
            if np.random.rand() < prob[bin_id[sample]]:
                sub_idx.append(idx_neurons[sample])

#     plt.hist(df_cell.loc[idx_neurons, 'side_pref_p'], binedges)
#     plt.hist(df_cell.loc[sub_idx, 'side_pref_p'], binedges)
#     plt.xlabel('Side selectivity')
#     plt.ylabel('cell count')
#     plt.legend(['Original', 'sub-selected'])

    drop_idx = [item for item in idx_neurons if item not in sub_idx]
    if len(drop_idx) != (np.max(num_cells) - np.min(num_cells)):
        raise Exception('um, something went wrong this is not supposed to happen')
    df_cell = df_cell.drop(drop_idx).reset_index()

    return df_cell



def plot_colorline(x,y,c, cmap = 'cividis'):
    import matplotlib.cm as cm

    c = cm.get_cmap(cmap)((c-np.min(c))/(np.max(c)-np.min(c)))
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=c[i])
    return


def reg_col(reg):
    if reg == 'DMS':
        return '#ca6641'
    elif reg == 'M2':
        return '#856798'
    
    
remove_nans = lambda M, axis: M[~np.isnan(M).any(axis=axis)]



def reconstruct_matrix(vec, mask): 
    
    # reconstructs a matrix from a vector which was generated 
    # by unraveling a matrix with NaNs and removing NaNs
    # takes mask as an input, which has nan at the same positions
    # as the original matrix, substitutes all non-nan values
    # with values from the vector
    
    reconst = np.empty(np.shape(mask))
    reconst.fill(np.nan)
    
    
    for tr in range(np.shape(mask)[0]):
        num_notzero = sum(~np.isnan(mask[tr,:]))
        reconst[tr, ~np.isnan(mask[tr,:])] = vec[:num_notzero]
        vec = vec[num_notzero:]
        
    return reconst



def xcorr(x, y, scale='unbiased', detrend=False, maxlags=None):
    # Cross correlation of two signals of equal length
    # scale sets how it is normalized (See matlab function xcorr)
    # Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)
    # Optional detrending e.g. mlab.detrend_mean

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')
    
    if detrend:
        import matplotlib.mlab as mlab
        x = mlab.detrend_mean(np.asarray(x)) # can set your preferences here
        y = mlab.detrend_mean(np.asarray(y))
    
    c = np.correlate(x, y, mode='full')

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                        'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    
    if scale == 'normed':
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)
    elif scale == 'unbiased':
        c = np.true_divide(c, x.size - abs(lags))
    elif scale == 'biased':
        c = np.true_divide(c, x.size)
    elif scale == 'None':
        c = c
    else:
        raise Warning("Don't understand the scale type")
    
    
    return lags, c



def flatten_list(t):
    flat_list = [item for sublist in t for item in sublist]
    return flat_list


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


