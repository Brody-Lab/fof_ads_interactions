import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from basis_kernels import make_kernel
from common_imports import to_t, from_t

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

class glm(nn.Module):

    def __init__(
            self,
            covar_dict,
            num_Y,
            dt=0.01,
            mean_function=nn.Softplus()):

        super(glm, self).__init__()
        self.num_Y = num_Y
        self.mean_function = mean_function
        self.covar_dict = covar_dict
        self.covariates = list(self.covar_dict)
        self.dt = dt

        # construct the GLM based on each entry in covariate dict
        for var in self.covariates:

            this_dict = self.covar_dict[var]
            
            # first add the kernels (convolutional bases - not fit)
            kernel_length = int(this_dict['duration']/self.dt)
            time_offset = int(np.abs(this_dict['time_offset'])/self.dt)
            setattr(self, var + '_basis', nn.Conv1d(
                in_channels = 1,
                out_channels = this_dict['num'],
                kernel_size = kernel_length,
                groups = 1,
                padding = kernel_length - 1 - time_offset,
                bias = False))
            
            kernels = make_kernel(this_dict, self.dt)
            reformat_kernel = np.flip(kernels.T, axis=1)[:, np.newaxis, :]
            getattr(self, var + '_basis').weight = nn.Parameter(to_t(reformat_kernel.copy()))
            getattr(self, var + '_basis').weight.requires_grad = False

            # then add linear weights
            setattr(self, var + '_w', nn.Linear(
                this_dict['num'],
                self.num_Y,
                bias=False))
            
        self.bias = nn.Parameter(torch.zeros(num_Y))
        
            
    def forward(self, X, Y):
        
        assert X.size(dim = 2) == len(self.covariates)
        num_time = Y.size(dim = 1)
        preds = 0.
        for i, var in enumerate(self.covariates):
            get_X = X[:,:,i][:,:,np.newaxis].permute(0,2,1)
            dm = getattr(self, var + '_basis')(get_X)[:,:,:num_time]
            preds += getattr(self, var + '_w')(dm.permute(0,2,1))
        
        preds += self.bias
        
        return self.dt * self.mean_function(preds)
    
    
    def make_state_dict(self):
        state_dict = dict()
        for name, param in self.named_parameters():
            if param.grad is not None:
                state_dict[name] = from_t(param)
                
        return state_dict
    
    
    
    
    
class glm_rr(nn.Module):

    def __init__(
            self,
            covar_dict,
            num_Y,
            rank_dict = None,
            dt=0.01,
            mean_function=torch.nn.Softplus()):

        super(glm_rr, self).__init__()
        self.num_Y = num_Y
        self.mean_function = mean_function
        self.covar_dict = covar_dict
        self.covariates = list(self.covar_dict)
        self.dt = dt
        self.rank_dict = rank_dict
        if self.rank_dict is not None:
            self.rank = self.rank_dict['rank']
        else:
            self.rank = 0

        # construct the GLM based on each entry in covariate dict
        for var in self.covariates:

            this_dict = self.covar_dict[var]
            
            # first add the kernels (convolutional bases - not fit)
            kernel_length = int(this_dict['duration']/self.dt)
            time_offset = int(np.abs(this_dict['time_offset'])/self.dt)
            setattr(self, var + '_basis', nn.Conv1d(
                in_channels = 1,
                out_channels = this_dict['num'],
                kernel_size = kernel_length,
                groups = 1,
                padding = kernel_length - 1 - time_offset,
                bias = False))
            
            kernels = make_kernel(this_dict, self.dt)
            reformat_kernel = np.flip(kernels.T, axis=1)[:, np.newaxis, :]
            getattr(self, var + '_basis').weight = nn.Parameter(to_t(reformat_kernel.copy()))
            getattr(self, var + '_basis').weight.requires_grad = False

            # then add linear weights
            setattr(self, var + '_w', nn.Linear(
                in_features = this_dict['num'],
                out_features = self.num_Y,
                bias=False))
            
            
        # then add rank related variables:
        if self.rank > 0:
            
            self.rr_U_w = nn.Linear(
                in_features = self.rank_dict['num_inputs'],
                out_features = self.rank,
                bias = False)
            self.norm_U_w = self.normalize_weights(self.rr_U_w, 1)
                
            self.rr_V_w = nn.Linear(
                in_features = self.rank,
                out_features = self.num_Y,
                bias = False)
            self.norm_V_w = self.normalize_weights(self.rr_V_w, 0)
            
            kernel_length = int(self.rank_dict['duration']/self.dt)
            self.rr_basis = nn.Conv1d(
                in_channels = self.rank,
                out_channels = self.rank * self.rank_dict['num'],
                kernel_size = kernel_length,
                groups = self.rank,
                padding = kernel_length - 1,
                bias = False)
            kernels = make_kernel(self.rank_dict, self.dt)
            reformat_kernel = np.flip(kernels.T, axis = 1)[:, np.newaxis, :]
            reformat_kernel = np.tile(reformat_kernel.T, self.rank).T
            self.rr_basis.weight = nn.Parameter(to_t(reformat_kernel.copy()))
            self.rr_basis.weight.requires_grad = False
            
            self.rr_basis_w = nn.Parameter(torch.randn(self.rank * self.rank_dict['num']))
                
        # then add a bias term
        self.bias = nn.Parameter(torch.zeros(num_Y))
        
            
    def forward(self, X, X_rank, Y):
        
        assert X.size(dim = 2) == len(self.covariates)
        num_time = Y.size(dim = 1)
        num_trials = Y.size(dim = 0)
        preds = 0.
        for i, var in enumerate(self.covariates):
            get_X = X[:,:,i][:,:,np.newaxis].permute(0,2,1)
            dm = getattr(self, var + '_basis')(get_X)[:,:,:num_time]
            preds += getattr(self, var + '_w')(dm.permute(0,2,1))
        
        if self.rank > 0:
            
            assert X_rank.size(dim = 2) == self.rank_dict['num_inputs']
            self.norm_U_w = self.normalize_weights(self.rr_U_w, 1)
            self.norm_V_w = self.normalize_weights(self.rr_V_w, 0)
            
            low_rank_proj = F.linear(X_rank, self.norm_U_w)
            
            # this is going to produce an output for each of basis kernels
            interaction = self.rr_basis(low_rank_proj.permute(0,2,1))[:,:,:num_time]
            
            # so we are going to sum them to get the net output from each kernel (mix with weights)
            interaction = torch.mul(interaction.permute(0,2,1), self.rr_basis_w)
            
            net_output = torch.sum(interaction.view(
                num_trials, 
                num_time,
                self.rank_dict['num'],
                self.rank), dim = 2)
    
            # and then project the activity back out
            preds += F.linear(net_output, self.norm_V_w)
        
        preds += self.bias
        
        return self.dt * self.mean_function(preds)
    
    
    def normalize_weights(self, linear_layer, dim):
        return F.normalize(linear_layer.weight, p = 2, dim = dim)
        # return linear_layer.weight
    
    def preds_rr(self, X, X_rank, Y):
        assert X.size(dim = 2) == len(self.covariates)
        num_time = Y.size(dim = 1)
        num_trials = Y.size(dim = 0)
        preds = 0.
        
        if self.rank > 0:
            
            assert X_rank.size(dim = 2) == self.rank_dict['num_inputs']
            self.norm_U_w = self.normalize_weights(self.rr_U_w, 1)
            self.norm_V_w = self.normalize_weights(self.rr_V_w, 0)
            
            low_rank_proj = F.linear(X_rank, self.norm_U_w)
            
            # this is going to produce an output for each of basis kernels
            interaction = self.rr_basis(low_rank_proj.permute(0,2,1))[:,:,:num_time]
            
            # so we are going to sum them to get the net output from each kernel (mix with weights)
            interaction = torch.mul(interaction.permute(0,2,1), self.rr_basis_w)
            
            net_output = torch.sum(interaction.view(
                num_trials, 
                num_time,
                self.rank_dict['num'],
                self.rank), dim = 2)
    
            # and then project the activity back out
            preds += F.linear(net_output, self.norm_V_w)
        
        
        return self.dt * self.mean_function(preds), net_output
        
    
    def make_state_dict(self):
        state_dict = dict()
        for name, param in self.named_parameters():
            if param.grad is not None:
                state_dict[name] = from_t(param)
                
        return state_dict
    
    
    
    

class glmDataset(Dataset):
    def __init__(self, X, Y):
        self.stimulus = to_t(X)
        
        # if only one neuron add another dim corr to neuron number
        if len(np.shape(Y)) == 2:
            Y = Y[:,:,np.newaxis]
        self.spikes = to_t(Y)
        
        

    def __len__(self):
        return self.stimulus.shape[0]

    def __getitem__(self, idx):
        # Binarize the stimulus, move it and the spikes to the GPU,
        # and package into a dictionary
        x = self.stimulus[idx].to(device).type(dtype) 
        y = self.spikes[idx].to(device)
        return dict(stimulus=x, spikes=y)
    
    
    
    
class glmRRDataset(Dataset):
    def __init__(self, X, X_pop, Y):
        self.stimulus = to_t(X)
        self.input_spikes = to_t(X_pop)
        self.spikes = to_t(Y)

    def __len__(self):
        return self.stimulus.shape[0]

    def __getitem__(self, idx):
        # Binarize the stimulus, move it and the spikes to the GPU,
        # and package into a dictionary
        x = self.stimulus[idx].to(device).type(dtype) 
        z = self.input_spikes[idx].to(device).type(dtype)
        y = self.spikes[idx].to(device)
        return dict(stimulus=x, input_spikes=z, spikes=y)
    
    

    
