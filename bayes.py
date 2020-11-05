'''
Here we define the abstract base class for Bayesian model.
Author: Benxin
Date: Jul. 24, 2020.
'''
import torch
import torch.nn as nn
import torch.nn.functional as nn_F

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import sys

from abc import ABC, abstractmethod

NOTEBOOK = 'ipykernel in sys.modules'

if 'tqdm' in sys.modules:
    if NOTEBOOK:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    HAS_TQDM = True
else:
    HAS_TQDM = False
    
    
class BayesModel(nn.Module, ABC):
    '''
    Abstract class, defining the sequential bayes model.
    '''
    def __init__(self, dim, tau=1, **kwargs):
        '''
        Function init:
        * Param(s):
            dim, int.      The dimension of feature.
            tau,   double. In interval [0, 1]. The forgetting ocefficient of S^{-1}.
                           If tau=0, all previous information will be dropped, and if
                           tau = 1, all previous information will be preserved.
                           (Default is 1).
            kwargs:
            mu,  np.ndarray/tensor, of shape (dim[+1], [1]). 
                         The initial mean vector of beta. If length is dim+1, the last 
                         value is the intercept.
            sig, np.ndarray/tensor, of shape (dim, dim), (dim+1, dim+1) or (dim[+1], [1]),
                 or double. 
                         The initial variance matrix of beta.
            sigma, double. The standard deviation of epsilon.
            a,     double. a := 1/(sigma)^2.
                         If a is given, sigma will be ignored.
            trace_loss, bool. Whether log the trace of loss.      
            trace_norm, bool. Whether log the trace of norm of S_i.
        '''
        super(BayesModel, self).__init__()
        self.dim = dim+1
        self.num_ite = 0
        
        assert tau>=0 and tau<=1
        self.tau_ = torch.tensor(tau).double()
        
        if 'trace_loss' in kwargs and kwargs['trace_loss']:
            self.trace_loss = True 
            self.func_loss = nn.MSELoss()
            self.lst_loss = []
        else:
            self.trace_loss = False
        if 'trace_norm' in kwargs and kwargs['trace_norm']:
            self.trace_norm = True
            self.lst_norm = []
        else:
            self.trace_norm = False
        if 'trace_mean' in kwargs and kwargs['trace_mean']:
            self.trace_mean = True
            self.lst_mean = []
        else:
            self.trace_mean = False
            
        # Initialise $mu, the mean value of beta.
        # If mu is not specified, we define it as 0, of shape (dim, 1).
        if 'mu' in kwargs:
            mu = kwargs['mu']
            if mu.shape == (self.dim, 1) or mu.shape == (self.dim, ):
                self.mu_ = torch.tensor(mu).double().reshape(self.dim, 1)
            elif mu.shape == (dim, 1) or mu.shape == (dim, ):
                mu = torch.tensor(mu).double().reshape(dim, 1)
                self.mu_ = torch.cat([mu, torch.zeros(1).double()], dim=0)
            else:
                raise TypeError(f'The shape of mu is expected to be {(self.dim, [1])} or {(dim, [1])}, while got {mu.shape}.')
        else:
            self.mu_ = torch.zeros(self.dim, 1).double()
            
        # Initialise $Sig, the variance matrix of beta.
        # If Sig is not specified, we define it as the identity matrix.
        if 'sig' in kwargs:
            sig = kwargs['sig']
            if isinstance(sig, np.ndarray) or isinstance(sig, torch.Tensor):
                if sig.shape == (self.dim, self.dim):
                    self.sig_ = torch.tensor(sig).double()
                elif sig.shape == (dim, dim):
                    sig = torch.tensor(sig).double()
                    sig = torch.cat([sig, torch.zeros(4,1).double()], dim=1)
                    tmp = torch.cat([torch.zeros(1,4).double(), torch.ones(1,1).double()], dim=1)
                    self.sig_ = torch.cat([sig, tmp], dim=0)
                elif sig.shape == (self.dim, ) or sig.shape==(self.dim, 1):
                    sig = torch.tensor(sig).double().reshape(-1)
                    self.sig_ = torch.diag(sig)
                elif sig.shape == (dim, ) or sig.shape==(dim, 1):
                    sig = torch.tensor(sig).double().reshape(-1, 1)
                    sig = torch.cat([sig, torch.ones(1,1).double()], dim=0)
                    self.sig_ = torch.diag(sig.reshape(-1))
        else:
            self.sig_ = torch.diag(torch.ones(self.dim).double())
                    
        self.sig_inv_ = torch.inverse(self.sig_)
        
        self.num_ite_old_ = 0
        self.mu_old_ = self.mu_
        self.sig_old_ = self.sig_
        self.sig_inv_old_ = self.sig_inv_
        
        
    @abstractmethod
    def forward(self, X):
        pass
    
    @abstractmethod
    def predict(self, X, **kwargs):
        pass
    
    @abstractmethod
    def update(self, Xi, Yi, **kwargs):
        pass
    
    @abstractmethod
    def fit(self, X, Y, **kwargs):
        pass
    
    def snapshot(self):
        '''
        Function used to keep a copy of current state.
        '''
        self.num_ite_old_ = self.num_ite
        self.mu_old_ = self.mu_
        self.sig_old_ = self.sig_
        self.sig_inv_old_ = self.sig_inv_
        
    def revert(self):
        '''
        Function used to revert the model to the stored state.
        Can be used to cancel the influence of one fit function.
        But can only keep one old state.
        '''
        self.num_ite = self.num_ite_old_
        self.mu_ = self.mu_old_
        self.sig_ = self.sig_old_
        self.sig_inv_ = self.sig_inv_old_
        
        if self.trace_loss:
            self.lst_loss = self.lst_loss[:self.num_ite]
        if self.trace_norm:
            self.lst_norm = self.lst_norm[:self.num_ite]
        if self.trace_mean:
            self.lst_mean = self.lst_mean[:self.num_ite]
            
    
    def plot_trace(self, start=None, end=None, **kwargs):
        '''
        Function used to plot the traced data.
        '''
        if start is None:
            start = 0
        if end is None:
            end = self.num_ite
            
        if self.trace_loss and self.trace_norm:
            fig, axs = plt.subplots(2,1, sharex=True)
            axs[0].plot(np.arange(start+1, end+1), self.lst_loss[start:end])
            axs[0].set_ylabel('Loss')
            axs[0].set_yscale('log')
            axs[0].set_title('Loss Trace')
            axs[1].plot(np.arange(start+1, end+1), self.lst_norm[start:end])
            axs[1].set_xlabel('Iteration')
            axs[1].set_ylabel('Norm')
            axs[1].set_yscale('log')
            axs[1].set_title('Norm Trace')
            fig.suptitle('Loss and Norm')

            if 'file' in kwargs:
                fig.savefig(kwargs['file'])
            plt.show()

            return

        elif self.trace_norm:
            fig, axs = plt.subplots()
            axs.plot(np.arange(start+1, end+1), self.lst_norm[start:end])
            axs.set_xlabel('Iteration')
            axs.set_ylabel('Norm')
            fig.suptitle('Norm Trace')

            if 'file' in kwargs:
                fig.savefig(kwargs['file'])
            plt.show()
        elif self.trace_loss:
            fig, axs = plt.subplots()
            axs.plot(np.arange(start+1, end+1), self.lst_loss[start:end])
            axs.set_xlabel('Iteration')
            axs.set_ylabel('Loss')
            fig.suptitle('Loss Trace')

            if 'file' in kwargs:
                fig.savefig(kwargs['file'])
            plt.show()
            
            
    def mean_trace(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = self.num_ite
        if self.trace_mean:
            return np.array(self.lst_mean[start:end]).reshape(-1, self.dim)
        
    def norm_trace(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = self.num_ite
        if self.trace_norm:
            return np.array(self.lst_norm[start:end])
    
    def loss_trace(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = self.num_ite
        if self.trace_loss:
            return np.array(self.lst_loss[start:end])
    