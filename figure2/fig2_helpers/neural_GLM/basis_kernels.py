import numpy as np

def make_raised_cosine_basis(num_bases = 5, 
                             peak_range = np.array([0, 0.8]), 
                             dt = 0.01, 
                             log_scaling = True, 
                             log_offset = .25, 
                             time_range = []):
    
    '''Makes basis of raised cosine with a logarithmic or linear time axis.
    Based on code by Jonathan Pillow: https://github.com/pillowlab/raisedCosineBasis.git
    Args:
        num_bases (Int): number of basis vectors
        peak_range (list with 2 elems): position of first and last cosine peaks
        dt (float): time bin size of bins representing basis
        log_scaling (bool): if True `log` otherwise `linear`. Default is `log` 
        log_offset (float > 0): offset for nonlinear stretching of t axis 
                    y = log(t+ log_offset) larger means more linear stretching
        time_range (np array with 2 elems): time range of basis
    Returns:
        cos_basis [num_bases x nT]: cosine basis vectors
        tgrid [nT]: time lattice on which basis is defined
        basis_peaks [num_bases x 1]: centers of each cosine basis function
    '''
    
    def raised_cos_fun(x, dc):
        x = x * (np.pi * (1 / dc) * 0.5)
        # set values greater than pi to pi
        x[x > np.pi] = np.pi
        x[x < -np.pi] = -1 * np.pi
        x = (np.cos(x) + 1) * 0.5  # rescale between 0 and 1
        return x
    
    peak_range = np.array(peak_range)
    
    if log_scaling is True:
        
        if log_offset <=0.:
            raise Exception("Ey! Don't feed log negative values, log_offset must be > 0 ")

        # defining nonlinear time axis function and its inverse
        nlin = lambda x: np.log(x + 1e-20)
        invnl = lambda x: np.exp(x - 1e-20)

        # computing location for cosine basis centers
        log_peak_range = nlin(peak_range + log_offset)  # first & last cosine peaks in stretched coordinates
        d_ctr = np.diff(log_peak_range)[0]/(num_bases-1) # spacing between raised cosine peaks
        B_ctrs = np.arange(log_peak_range[0], 
                           log_peak_range[1] + d_ctr, 
                           d_ctr)[:num_bases]  #peaks for cosine basis vectors
        B_ctrs = B_ctrs.reshape([num_bases, 1])

        # compute time grid points
        if len(time_range) == 0:
            minT = 0 # minimum time bin (where first basis function starts)
            maxT = invnl(log_peak_range[1] + 2*d_ctr) - log_offset # maximum time bin (where last basis function stops)
        else:
            minT, maxT = time_range[0], time_range[1]
                
        tgrid = np.arange(minT, maxT+dt, dt)  # time grid
        nT = tgrid.shape[0]
        tgrid_rep = np.tile(nlin(tgrid + log_offset), [num_bases, 1])
        B_ctrs_rep = np.tile(B_ctrs, [1, nT])
        cos_basis = raised_cos_fun(tgrid_rep - B_ctrs_rep, d_ctr)
        
        
    elif log_scaling is False:
        
        d_ctr = np.diff(peak_range)[0]/(num_bases-1) # spacing between raised cosine peaks
        B_ctrs = np.arange(peak_range[0],  # peaks for cosine basis vectors
                           peak_range[1] + d_ctr, 
                           d_ctr).reshape([num_bases, 1])   
        if len(time_range) == 0:
            minT = peak_range[0] - 2*d_ctr # min time bin (where 1st basis vector starts)
            maxT = peak_range[1] + 2*d_ctr # max time bin (where last basis vector stops)
        else:
            minT, maxT = time_range[0], time_range[1]
        
        tgrid = np.arange(minT, maxT+dt, dt)  # time grid
        nT = tgrid.shape[0]
        tgrid_rep = np.tile(tgrid, [num_bases, 1])
        B_ctrs_rep = np.tile(B_ctrs, [1, nT])
        cos_basis = raised_cos_fun(tgrid_rep - B_ctrs_rep, d_ctr)
        
        
    cc = np.linalg.cond(cos_basis)
    if cc > 1e12:
        raise Exception("Raised cosine basis is poorly conditioned (cond # =" + str(cc) + ")")
        
    return cos_basis.T, tgrid


# basis, tgrid = make_raised_cosine_basis(
#     num_bases=10,
#     peak_range=np.array([0.1, 1.6]),
#     dt=0.001,
#     log_scaling=True,
#     log_offset=0.1,
#     time_range=[0, 2.5])

# plt.plot(tgrid, basis);



def make_kernel(this_dict, dt):
    
    kernel_length = int(this_dict['duration']/dt)
    kernels, _ = make_raised_cosine_basis(
        num_bases=this_dict['num'],
        peak_range=this_dict['peak_range'],  # in s
        dt=dt,
        log_scaling=this_dict['log_scaling'],
        log_offset=this_dict['log_offset'],
        time_range=[dt, kernel_length*dt])
    
    return kernels
    