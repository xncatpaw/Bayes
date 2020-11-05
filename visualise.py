'''
This module defines functions used to visualise a random variable or a 2-dim random vector.
Developped by Benxin, 16, Juillet, 2020.
'''

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def visualise(X, **kwargs):
    '''
    Function used to plot the pdf of a random variable, or of a 2-dim random vector.
    * Param(s): 
        X,     scipy.stats.rv_continuous/frozen-multivariate normal 
                    The random variable/2-dim random vector to visualise.
        **kwargs:
        area,  array-like. Of shape (2,), indicating [min, max] or of shape (2, 2),
                    indicating [[min_1, max_1], [min_2, max_2]].
                    The range in which the pdf is plotted. If it has shape (2, ) and 
                    dim is 2, a square arrea will be used. If this param is not specified,
                    [mu-3std, mu+3std] will be used.
                    
    '''
    # Determine the dim of variable.
    if callable(X.mean):  # X is a rv_continuous
        mu = X.mean()
        var = X.var()
    else:
        mu = X.mean
        var = X.cov
    dim = np.array(mu).size
    assert dim <= 2
    
    if 'area' in kwargs:
        area = np.array(kwargs['area'])
        if dim==1 and area.size==4:
            area = area[0]
        if dim==2 and area.size==2:
            area = np.array([area, area])
    else:
        if dim == 1:
            std = np.sqrt(var)
            area = np.array([mu-3*std, mu+3*std])
        else:
            std = np.sqrt(np.diag(var))
            area = np.array([mu-3*std, mu+3*std]).transpose()
    
    
    if dim==1:
        x = np.linspace(area[0], area[1], 100).flatten()
        fig, ax = plt.subplots()
        ax.plot(x, X.pdf(x).flatten())
        fig.legend(['pdf'])
        fig.show()
    else:
        x = np.linspace(area[0,0], area[0,1], 100)
        y = np.linspace(area[1,0], area[1,1], 100)
        x, y = np.meshgrid(x, y)
        pos = np.dstack((x, y))
        
        z = X.pdf(pos)
        
        fig = plt.figure()
        if '_3d' in kwargs and kwargs['_3d']:
            ax = fig.gca(projection='3d')
            ax.plot_surface(x, y, z, cmap=cm.viridis)
            cset = ax.contourf(x, y, z, zdir='z', offset=-.15, cmap=cm.viridis)
            ax.set_zlim(-.15, np.max(z))
            fig.colorbar(cset)
        
        else:
            ax = fig.gca()
            cset = ax.contourf(x, y, z, cmap=cm.viridis)
            fig.colorbar(cset)
        
        if 'not_show' in kwargs and kwargs['not_show']:
            return fig, ax
        else:
            fig.show()
        
        
        
    
    