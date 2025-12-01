import numpy as np
import torch
from basis_kernels import make_kernel

from common_imports import *
from figure2.fig2_helpers.DVcc_sims import PoissonClicks

# construct X with three regressor clicks and choice
def simulations_get_X(num_trials, dt, total_time = 1.5):

    # for simulations
    params = dict()
    params['dt'] = dt
    params['history_bias'] = True
    params['T'] = [0.2, 1.0]
    params['N_batch'] = num_trials
    pc_task = PoissonClicks(**params)
    left_clicks, right_clicks, task_data = pc_task.get_trial_batch(get_LR=True)

    # click regressors
    nan_index = np.isnan(left_clicks)
    left_clicks[nan_index] = 0.
    right_clicks[nan_index] = 0.

    # assign choice to each trial
    def psych(x, h): return 0.1 + 0.8*1/(1+np.exp(-0.1*x + 0.5*h))
    pchooseR = [psych(x, h) for (x, h) in zip(
        task_data['Δclicks'], task_data['history_bias'])]
    choice = np.random.rand(len(pchooseR)) < pchooseR
    choice = 2 * choice - 1
    choice_time = np.round(
        np.array(task_data['T'])/dt).astype(int) + np.random.choice(int(0.3/dt), (len(choice)))
    

    # okay now make X:
    X = np.zeros((num_trials, int(total_time/dt), 3))
    X[:, :left_clicks.shape[1], 0] = left_clicks
    X[:, :right_clicks.shape[1], 1] = right_clicks
    for i in range(num_trials):
        X[i, choice_time[i], 2] = choice[i]

    return X



def simulations_get_weights(covar, num_Y, dt):

    for var in covar:
        this_dict = covar[var]
        this_dict['weights'] = np.random.normal(
            loc=0,
            scale=1.,
            size=((this_dict['num'], num_Y)))
        this_dict['kernels'] = make_kernel(this_dict, dt)

    # bias for each neuron
    b = np.random.choice(10, num_Y)
    
    return covar, b


def simulations_get_rank_weights(rank_dict, num_Y, dt):
     
    rank = rank_dict['rank']
    
    # Get vectors for low rank projection
    U = np.random.normal(
        loc = 0,
        scale = 1.,
        size = (rank, rank_dict['num_inputs']))
    U = gram_schmidt(U)
    rank_dict['U'] = normalize_vectors(U).T
    
    # Get vectors for projecting back up
    V = np.random.normal(
        loc = 0.,
        scale = 1.,
        size = (rank, num_Y))
    V = gram_schmidt(V).T
    rank_dict['V'] = normalize_vectors(V).T 
    
    rank_dict['weights'] = np.random.normal(
        loc = 0,
        scale = 4.,
        size = rank_dict['rank'] * rank_dict['num'])
    
    rank_dict['kernels'] = make_kernel(rank_dict, dt)

    return rank_dict


def simulations_get_Y(X, covar, b, num_Y, rank_dict = None, Y_input = None, dt = 0.001):
    
    Y_rate = np.zeros((X.shape[0], X.shape[1], num_Y))
    
    for i, var in enumerate(covar):
        this_dict = covar[var]
        for nt in range(X.shape[0]):
            for y in range(num_Y):
                this_kernel = np.dot(this_dict['weights'][:,y], this_dict['kernels'].T)
                Y_rate[nt,:,y] += np.convolve(X[nt,:,i], this_kernel)[:X.shape[1]]

    if rank_dict is not None and Y_input is not None:
        Y_rate += simulations_get_rr_Y(X, num_Y, rank_dict, Y_input, dt)
                
    Y_rate += b
    sp = torch.nn.Softplus()
    Y_rate = sp(to_t(Y_rate))
    Y = from_t(torch.poisson(Y_rate * dt))
    
    return Y, from_t(Y_rate)



def gram_schmidt(vectors):
    num_vecs = len(vectors)
    ortho_basis = np.zeros_like(vectors)
    for i in range(num_vecs):
        # Orthogonalize the vector against previous vectors
        new_vector = vectors[i]
        for j in range(i):
            new_vector -= np.dot(ortho_basis[j], vectors[i]) / np.dot(ortho_basis[j], ortho_basis[j]) * ortho_basis[j]
        ortho_basis[i] = new_vector
    return ortho_basis


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1)
    return vectors / norms[:, np.newaxis]


def simulations_get_rr_Y(X, num_Y, rank_dict, Y_input, dt = 0.001):
    
    Y_rate = np.zeros((X.shape[0], X.shape[1], num_Y))
      
    # from the other neural population
    rank = rank_dict['rank']
    low_rank = np.dot(Y_input, rank_dict['U'])
    this_kernel = []
    for r in range(rank):
        idx0 = r*rank_dict['num']
        idx1 = (r+1)*rank_dict['num']
        this_kernel.append(np.dot(rank_dict['weights'][idx0:idx1], rank_dict['kernels'].T))
        
    for r in range(rank):
        for nt in range(X.shape[0]):
            this_input = np.convolve(low_rank[nt,:,r], this_kernel[r])[:X.shape[1]][:,np.newaxis]
            Y_rate[nt,:,:] += this_input @ rank_dict['V'][r][np.newaxis,:] 
    
    return Y_rate

