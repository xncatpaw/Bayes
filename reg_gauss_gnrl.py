'''
In this file, we give the implementation of a generalised bayesian regression model 
with mean-field variational bayes method.
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
    
    
class RegGaussGnrl(BayesModel):
    '''
    The class that defines the generalised bayesian regression model.
    We consider the error of linear regression as a Gaussian distribution 
    centred at 0 and the convariance is unkown. And that the coeficient
    vecotr is a random gaussian vector.
    '''
    
    def __init__(self, dim, tau=1, kappa=1, tol=.01, **kwargs):
        '''
        Init function.
        * Param(s):
            dim,   int.    The dimension of feature.
            tau,   double. In interval [0, 1]. The forgetting ocefficient of S^{-1} and m.
                           If tau=0, all previous information will be dropped, and if
                           tau = 1, all previous information will be preserved.
                           (Default is 1).
            kappa, double. In interval [0, 1]. The forgetting ocefficient of a and b.
                           If tau=0, all previous information will be dropped, and if
                           tau = 1, all previous information will be preserved.
                           (Default is 1).
            tol,   double. The tolerance for MFVB interation. (Default is 0.01).
            kwargs:
            mu,  np.ndarray/tensor, of shape (dim[+1], [1]). 
                         The initial mean vector of beta. If length is dim+1, the last 
                         value is the intercept.
            sig, np.ndarray/tensor, of shape (dim, dim), (dim+1, dim+1) or (dim[+1], [1]),
                 or double. 
                         The initial variance matrix of beta.
            a,     double. The initial hyperparam a in the Gamma(a, b) dsstribution of 
                         alpha. If a is not given, a=2 is used.
            b,     double. The initial hyperparam b in the Gamma(a, b) distribution of 
                         alpha. If b is not given, b=2 is used.
            trace_loss, bool. Whether log the trace of loss.      
            trace_norm, bool. Whether log the trace of norm of S_i.
            trace_mean, bool. Whether log the trace of mean of beta.
            trace_a,    bool. Whether log the trace of a
            trace_b,    bool. Whether log the trace of b
        '''
    
        super(RegGaussGnrl, self).__init__(dim, tau, **kwargs)
        self.kappa_ = kappa
    
        if 'trace_a' in kwargs and kwargs['trace_a']:
            self.trace_a = True
            self.lst_a = []
        else:
            self.trace_a = False
        if 'trace_b' in kwargs and kwargs['trace_b']:
            self.trace_b = True
            self.lst_b = []
        else:
            self.trace_b = False
            
        self.tol_ = tol
        
        if 'a' in kwargs:
            assert(kwargs['a'] > 0)
            self.a_ = torch.tensor(kwargs['a']).double()
        else:
            self.a_ = torch.tensor(2.0).double()
        if 'b' in kwargs:
            assert(kwargs['b'] > 0)
            self.b_ = torch.tensor(kwargs['b']).double()
        else:
            self.b_ = torch.tensor(2.0).double()
            
        self.a_old_ = self.a_
        self.b_old_ = self.b_
        
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
            
        * Return:
            Y|X, a scipy random variable or random vector.
                 If 'indiv' is specified as True, the return value is a 
                 sequence of random variables: [y_1|x_1, ..., y_n|x_n]^t.
        '''
        if 'method' in kwargs:
            method = kwargs['method']
            assert method==1 or method==2
        else:
            method = 2
        
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).double()
        X = X.reshape(-1, self.dim-1)
        
        n = len(X)
        X = torch.cat([X, torch.ones(n,1).double()], dim=1)    
        
        with torch.no_grad():
            # If we use approach 1.
            if method == 1:
                mu_Y = torch.mm(X, self.mu_)
                tmp = torch.mm(self.sig_, X.transpose(0,1)) # SX^t
                tmp = torch.mm(X, tmp) # XSX^t
                S_Y =  self.b_/self.a_*torch.diag(torch.ones(n).double()) + tmp
                
            # Approach 2.
            else:
                alpha = self.a_/self.b_
                alpha_sqr = (self.a_**2 + self.a_)/(self.b_**2)
                
                # Compute S1.
                S0_inv = self.sig_inv_
                S1_inv = S0_inv + alpha*torch.mm(X.transpose(0,1), X)
                S1 = torch.inverse(S1_inv)
                
                # Compute S_Y, i.e. \Sigma_{Y|X}.
                tmp = torch.mm(S1, X.transpose(0,1)) # S_1X^t.
                tmp = torch.mm(X, tmp) # XS_1X^t.
                S_Y_inv = alpha*torch.diag(torch.ones(n).double()) - alpha_sqr*tmp #\alpha I - \alpha^2 XS_1X^t
                S_Y = torch.inverse(S_Y_inv)
                
                # Compute mu_Y.
                tmp = torch.mm(S0_inv, self.mu_) # S^{-1}m
                tmp = torch.mm(S1, tmp) # S_1 S^{-1} m
                tmp = torch.mm(X, tmp) # X S_1 S^{-1} m
                tmp = torch.mm(S_Y, tmp) # \Sigma_{Y|X} X S_1 S^{-1} m
                mu_Y = alpha * tmp
            
            # Build the random variable.    
            if n==1:
                assert mu_Y.shape==(1,1)
                assert S_Y.shape==(1,1)
                sigma_Y = torch.sqrt(S_Y)
                Y = stats.norm(loc=mu_Y, scale=sigma_Y)
            else:
                mu_Y = mu_Y.reshape(-1)
                Y = stats.multivariate_normal(mean=mu_Y, cov=S_Y)

            return Y
                
                    
        
        
        
#     @overrides(BayesModel)
    def snapshot(self):
        super(RegGaussGnrl, self).snapshot()
        self.a_old_ = self.a_
        self.b_old_ = self.b_
#     @overrides(BayesModel)
    def revert(self):
        super(RegGaussGnrl, self).revert()
        self.a_ = self.a_old_
        self.b_ = self.b_old_
        
        if self.trace_a:
            self.lst_a = self.lst_a[:self.num_ite]
        if self.trace_b:
            self.lst_b = self.lst_b[:self.num_ite]
        
#     @overrides(BayesModel)
    def update(self, Xi, Yi, **kwargs):
        '''
        Function used to update the model.
        * Param(s):
            Xi, tensor/np.array.  Of shape (n, d).
            Yi, tensor/np.array.  Of shape (n, 1).
        The formulas are:
            a_{i+1} = a_{i} + N/2,
            b_{i+1} = b_{i} + C_{i+1}/2,
            m_{i+1} = S_{i+1}(mu_{i+1}X^tY + S_{i}^{-1}m_i)
            S_{i+1} = (S_{i}^{-1} + mu_{i+1}X^tY)^{-1}
        where
            C_{i} = (Y-Xm_i)^t(Y-Xm_i) + Tr(XS_iX^t). 
        '''
        if not isinstance(Xi, torch.Tensor):
            Xi = torch.tensor(Xi).double()
        if not isinstance(Yi, torch.Tensor):
            Yi = torch.tensor(Yi).double()
            
        Xi_orig = Xi
        
        if 'tol' in kwargs:
            tol = kwargs['tol']
        else:
            tol = self.tol_
        if 'max_it' in kwargs:
            max_it = kwargs['max_it']
        else:
            max_it = np.Inf
        if 'p' in kwargs:
            p = kwargs['p']
        else:
            p = 2
        
        n = len(Xi)
        assert len(Yi) == n
        
        if self.trace_loss:
            func_loss = nn.MSELoss()
            
        with torch.no_grad():
            S0_inv = self.sig_inv_ #* self.tau_
            m0 = self.mu_ #* self.tau_
            a0 = self.a_ #* self.tau_
            b0 = self.b_ #* self.tau_
            # Append the list for intercept.
            Xi = torch.cat([Xi, torch.ones(n,1).double()], dim=1)
            a_new = a0*self.kappa_ + n/2 # New a value.
            b = self.b_
            m = self.mu_
            S_inv = self.sig_inv_
            
            # The interation process
            chg = np.Inf # Initial change value.
            num_it = 0   # Number of interation
            while chg > tol and num_it < max_it:
                b_new, m_new, S_new_inv = self.mfvb_it_(a_new, b, m, S_inv, Xi, Yi)
                chg = RegGaussGnrl.dist(b, m, S_inv, b_new, m_new, S_new_inv, p=p)
                b = b_new
                m = m_new
                S_inv = S_new_inv
                num_it += 1
            
            # Update 
            self.a_ = a_new
            self.b_ = b
            self.mu_ = m
            self.sig_inv_ = S_inv
            self.sig_ = torch.inverse(self.sig_inv_) 
        
            # Save trace
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
            if self.trace_a:
                self.lst_a.append(self.a_.numpy())
            if self.trace_b:
                self.lst_b.append(self.b_.numpy())
                
        self.num_ite += 1
        if 'verbose' in kwargs and kwargs['verbose']:
            print(f'Iterated {num_it} times. Final change is {chg}.')
            
                
            
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
            kappa,      double. In the interval [0, 1]. Used to replace self.kappa_ 
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
        
        
#     @staticmethod
    def dist(*args, **kwargs):
        L = len(args)
        assert(L%2 == 0) # Must have even number of args.
        
        if 'p' in kwargs:
            p = kwargs['p']
        else:
            p = 2
        
        dist = 0
        m = int(L/2)
        for i in range(m):
            x0 = args[i]
            x1 = args[i+m]
            dist = dist + torch.norm(x0-x1, p)
            
        return dist
        
    def mfvb_it_(self, a, b, m, S_inv, X, Y):
        '''
        Function defined for one interation of BFVB.
        '''
#         N = len(X)
        S0_inv = self.sig_inv_
        m0 = self.mu_
        b0 = self.b_
        
#         a = self.a_ + N/2
        mu_alpha = a / b # The mean of alpha, used to calculate S_{i+1} and m_{i+1}.
        # Compute S_{i+1}
        S_new_inv = S0_inv*self.tau_ + mu_alpha*torch.mm(X.transpose(0,1), X) # The inverse of S_{i+1}.
        S_new = torch.inverse(S_new_inv)
        # Compute m_{i+1}
        tmp = mu_alpha*torch.mm(X.transpose(0,1), Y) + torch.mm(S0_inv, m0*self.tau_) # mu_{i+1}X^tY + S_{i}^{-1}m_i
        m_new = torch.mm(S_new, tmp)
        # Compute C_{i+1}
        tmp = Y - torch.mm(X, m_new) # Y - Xm_{i+1}
        tmp_2 = torch.mm(S_new, X.transpose(0,1)) # S_{i+1}X^t
        tmp_2 = torch.mm(X, tmp_2) # XS_{i+1}X^t
#         print('tmp2 shape: ', tmp_2.shape)
        C = torch.dot(tmp.view(-1), tmp.view(-1)) + torch.trace(tmp_2)
        # Compute b_{i+1}
        b_new = b0*self.kappa_ + C/2
        
        return b_new, m_new, S_new_inv
    
    
    
    def get_alpha(self):
        alpha = stats.gamma(a=self.a_, scale=(1/self.b_))
        return alpha
    
    def get_beta(self):
        mu = self.mu_.numpy().reshape(-1)
        cov = self.sig_
        beta = stats.multivariate_normal(mean=mu, cov=cov)
        
        return beta
        
        
    def a_trace(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = self.num_ite
        if self.trace_a:
            return np.array(self.lst_a[start:end]).reshape(-1)
    def b_trace(self, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = self.num_ite
        if self.trace_b:
            return np.array(self.lst_b[start:end]).reshape(-1)
        