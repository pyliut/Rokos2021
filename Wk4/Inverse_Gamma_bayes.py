# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:02:06 2021

@author: pyliu
"""
import numpy as np
import scipy as sp

from Inverse_Digamma_bounds import *
from Inverse_Digamma_newton import *

def Inverse_Gamma_bayes(x, a, b, c, d, e, tol = 0.001):
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
    
    #transform hyperparameters of both single-parameter conjugate priors
    e = e + np.sum( 1/x )
    log_a = np.log(a) + np.sum( np.log(x) )
    b = b + n
    c = c + n
    log_e = np.log(e)
    
    #Inverse digamma
    error = 1000
    counter = 0
    while error > tol:
        alpha_prev = alpha
        k = ( -log_a + c*(np.log(d + n*alpha_prev) - log_e) ) / b
        alpha = Inverse_Digamma_bounds(k)
        error = np.abs(alpha_prev - alpha)
        counter += 1
    print("count:", counter)
        
    
    #Estimate beta (param of Invgamma)
    d = d + n*alpha
    beta = d/e
    
    return alpha, beta