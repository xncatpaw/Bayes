'''
Here we implement the bayesian gaussian linear regression model.
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

from .bayes import BayesModel

NOTEBOOK = 'ipykernel in sys.modules'

if 'tqdm' in sys.modules:
    if NOTEBOOK:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    HAS_TQDM = True
else:
    HAS_TQDM = False
    

class RegGauss(BayesModel):
    '''
    The class which defines the sequential-bayesian linear-regression model.
    In this model, the priority and posterier are gaussian.
    Check coding/project_log.md for more information.
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
            alpha, double. alpha := 1/(sigma)^2.
                         If alpha is given, sigma will be ignored.
            trace_loss, bool. Whether log the trace of loss.      
            trace_norm, bool. Whether log the trace of norm of S_i.
            trace_mean, bool. Whether log the trace of mean of beta.
        '''
        super(RegGauss, self).__init__(dim, tau, **kwargs)
        
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        elif 'sigma' in kwargs:
            sigma = kwargs['sigma']
            self.alpha = 1/(sigma*sigma)
        else:
            self.alpha = 1
        
#     @overrides(BayesModel)    
    def forward(self, X):
        '''
        Forward function. Override of nn.Module.forwad.
        * Param(s):
            X, np.ndarray/tensor, of shape (n_sample, dim) or (dim, ).
        * Return:
            Y, tensor, of shape (n_sample, 1) or (1,).
                The linear regression fitted value.
        '''
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        assert X.shape==(self.dim-1, ) or X.shape[1]==self.dim-1
        with torch.no_grad():
            X = X.reshape(-1, self.dim-1).double()
            n = len(X)
            X = torch.cat([X, torch.ones(n,1).double()], dim=1)
            
            Y = torch.mm(X, self.mu_)
            
            return Y
        
        
#     @overrides(BayesModel)
    def predict(self, X, **kwargs):
        '''
        Function used to generate the conditional random variable Y|X.
        * Param(s):
            X, np.ndarray/tensor, of shape (n_smaple, dim) or (dim,).
            kwargs:
            indiv, bool.  Whether calculate each y_i|x_i individually.
        * Return:
            Y|X, a scipy random variable or random vector.
                 If 'indiv' is specified as True, the return value is a 
                 sequence of random variables: [y_1|x_1, ..., y_n|x_n]^t.
        '''
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).double()
        X = X.reshape(-1, self.dim-1)
        
        n = len(X)
        X = torch.cat([X, torch.ones(n,1).double()], dim=1)    
        alpha = self.alpha
        
        # \mu_{y|x} = \frac{<x, S^{-1}m>_{S_1}}{1-a<x,x>_{S_1}}.
        # \frac{1}{\sigma_{y|x}^2} = \frac{1}{\sigma^2} - <\frac{x}{\sigma^2}, \frac{x}{\sigma^2}>_{S_1}.
        # For multi-dim, 
        # \Sigma_{Y|X}^{-1} = aI_n - a^2XS_1X^t.
        # \mu_{Y|X} = a\Sigma_{Y|X}XS_1S^{-1}m.
        # Check coding/project_log.md : Bayesian Regression for more information.
        
        with torch.no_grad():
            # S_1
            S0_inv = self.sig_inv_
            S1_inv = alpha*torch.mm(X.transpose(0,1), X) + S0_inv
            S1 = torch.inverse(S1_inv)

            # \Sigma_{Y|X}
            tmp = torch.mm(S1, X.transpose(0,1)) # S1X, tmp value.
            SY_inv = alpha*torch.diag(torch.ones(n)).double() - (alpha**2)*torch.mm(X, tmp)
            SY = torch.inverse(SY_inv)

            # \mu_{Y|X}
            tmp = torch.mm(S0_inv, self.mu_) # S^{-1}m.
            tmp = torch.mm(S1, tmp) # S_1S^{-1}m.
            tmp = torch.mm(X, tmp) # XS_1S^{-1}m.
            tmp = torch.mm(SY, tmp) # \Sigma_{Y|X} XS_1S^{-1}m.
            mu_Y = alpha * tmp  
            # Rmk: This order of multiplication is optimal, with total time complexity O(n^2d + nd^2 + 2d^2).

            # If just one x.
            if n==1: 
                assert SY.shape==(1,1)
                assert mu_Y.shape==(1,1)
                sigma_Y = torch.sqrt(SY)
                Y = stats.norm(loc=mu_Y, scale=sigma_Y)
                
                return Y
            else:
                
                if 'debug' in kwargs and kwargs['debug']:
                    print(SY)
                
                mu_Y = mu_Y.reshape(-1)
                Y = stats.multivariate_normal(mean=mu_Y, cov=SY)
                
                return Y
        
        
        
#     @overrides(BayesModel)
    def update(self, Xi, Yi, **kwargs):
        '''
        Function used to update the model.
        '''
        if not isinstance(Xi, torch.Tensor):
            Xi = torch.tensor(Xi).double()
        if not isinstance(Yi, torch.Tensor):
            Yi = torch.tensor(Yi).double()
            
        Xi_orig = Xi
        
        n = len(Xi)
        assert len(Yi) == n
        
#         if self.trace_loss:
#             func_loss = nn.MSELoss()
        
        with torch.no_grad():
            # Append the list for intercept.
            Xi = torch.cat([Xi, torch.ones(n,1).double()], dim=1)
            # S_{i+1}^{-1} = aX_i^tX_i + S_{i}^{-1}. 
            A = torch.mm(Xi.transpose(0,1), Xi)
            sig_inv = self.alpha*A + self.sig_inv_*self.tau_ # With forgetting coef.
            # Update the sig_ matrix as S_{i+1}, but keep sig_inv_ unchanged.
            self.sig_ = torch.inverse(sig_inv) 
            
            # Here self.sig_inv_ is still S_{i}^{-1}, but NOT S_{i+1}^{-1}.
            # And self.mu_ is still mu_{i}.
            # m_{i+1} = S_{i+1}(aX_i^t Y_i + S_{i}^{-1}m_{i}).
            mu_old = self.mu_*self.tau_
            mu = self.alpha*torch.mm(Xi.transpose(0,1), Yi) + torch.mm(self.sig_inv_, mu_old) 
            mu = torch.mm(self.sig_, mu)
            self.mu_ = mu
            # Here we update the sig_inv_ matrix as S_{i+1}^{-1}.
            self.sig_inv_ = sig_inv
            
            if self.trace_loss:
                Yi_p = self.forward(Xi_orig)
                loss = self.func_loss(Yi, Yi_p)
                self.lst_loss.append(loss)
            if self.trace_norm:
                eigvals = np.linalg.eigvals(self.sig_)
                norm = np.max(eigvals)
                self.lst_norm.append(norm)
            if self.trace_mean:
                self.lst_mean.append(self.mu_.numpy())
                
        
        self.num_ite += 1 # Add the iteration number by 1.
        
#     @overrides(BayesModel)
    def fit(self, X, Y, chunck_size=None, verbose=False, **kwargs):
        '''
        Function used to fit the model.
        Remark: this function will NOT re-initialise the model.
        * Parma(s):
            X, np.ndarray/tensor, of shape (n_smaple, dim).
            Y, np.ndarray/tensor, of shape (n_sample, [1]).
            chunck_size, int. The size of each data chunck for iteration.
                              (Default is n_sample).
            kwargs:
            plot,       bool. Whether plot the change of loss or the change of norm.
                              If $trace_loss is False, this param will be ignored.
            tau,        double. In the interval [0, 1]. Used to replace self.tau_ 
                              TEMPORARILY. 
                              
        '''
        
        self.snapshot()
        
        n_sample = len(X)
        assert len(Y) == n_sample
        X = torch.tensor(X).double()
        Y = torch.tensor(Y).double().reshape(-1, 1)
        if chunck_size is None:
            chunck_size = n_sample
        
        
        lst_it = list(range(0, n_sample, chunck_size))
        if verbose and HAS_TQDM:
            lst_it = tqdm(lst_it)
            
        init_num_ite = self.num_ite
        
        # If we want to specify the forgetting coefficient.
        if 'tau' in kwargs:
            tau = kwargs['tau']
            assert tau>=0 and tau<=1
            origin_tau = self.tau_
            self.tau_ = tau
            
        with torch.no_grad():
            for i in lst_it:
                if i+chunck_size < n_sample:
                    Xi = X[i:i+chunck_size]
                    Yi = Y[i:i+chunck_size]
                else:
                    Xi = X[i:]
                    Yi = Y[i:]
                    
                self.update(Xi, Yi) # Update the model.
        if 'tau' in kwargs:
            self.tau_ = origin_tau
            
        if ('plot' in kwargs and kwargs['plot']):
            self.plot_trace(start=init_num_ite, **kwargs)
            
            
    def get_beta(self):
        mu = self.mu_.numpy().reshape(-1)
        cov = self.sig_
        beta = stats.multivariate_normal(mean=mu, cov=cov)
        
        return beta