# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 11:09:03 2021

@author: pyliu
"""

import numpy as np
import scipy as sp

def Inverse_Gamma_bayes2(x, w1, w2, tol = 0.001):
    """
    

    Parameters
    ----------
    x : FLOAT, vector
        observed durations
    a : FLOAT, scalar
        Hyperparameter of prior
    b : FLOAT, scalar
        Hyperparameter of prior
    c : FLOAT, scalar
        Hyperparameter of prior
    d : FLOAT, scalar
        Hyperparameter of prior
    e : FLOAT, scalar
        Hyperparameter of prior
    tol : FLOAT, scalar
        Minimum difference between successive estimations of alpha. The default is 0.001.

    Returns
    -------
    alpha : FLOAT, scalar
        Parameter of invgamma distribution
    beta : FLOAT, scalar
        Parameter of invgamma distribution

    """
    n = len(x)
    
    #mean & variance MLE to start
    mu = (1/n) * np.sum(x)
    nu = (1/(n-1)) * np.sum( np.square(x-mu) )
    
    #initial estimate of alpha (param of Invgamma)
    alpha = (mu**2)/nu + 2
    
    error = 1000
    counter = 0
    while error > tol:
        alpha_prev = alpha
        k1 = n*( np.mean( -np.log(x) ) - sp.special.digamma(alpha) + np.log(n*alpha) - np.log(np.sum(1/x)) - alpha * sp.special.polygamma(1,alpha) + 1)
        k2 = n * (alpha**2 * sp.special.polygamma(1,alpha) - alpha)
        w1 = w1 + k1
        w2 = w2 + k2
        alpha = -w2/w1
        error = np.abs(alpha_prev - alpha)
        counter += 1
    print("count:", counter)
        
    beta = n*alpha / np.sum(1/x)
    
    return alpha, beta
    
    
        
        
        
        