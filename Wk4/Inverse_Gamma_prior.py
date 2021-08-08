# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:58:28 2021

@author: pyliu
"""
import numpy as np
import scipy as sp

def Inverse_Gamma_prior(alpha, beta, a, b, c, d, e):
    """
    Prior for Inverse Gamma pdf with both variables unknown

    Parameters
    ----------
    alpha : FLOAT, N-dim vector
        parameter of Inverse Gamma distribution
    beta : FLOAT, N-dim vector
        parameter of Inverse Gamma distribution
    a : FLOAT, scalar
        parameter of prior for Inverse Gamma 
    b : FLOAT, scalar
        parameter of prior for Inverse Gamma 
    c : FLOAT, scalar
        parameter of prior for Inverse Gamma 
    d : FLOAT, scalar
        parameter of prior for Inverse Gamma 
    e : FLOAT, scalar
        parameter of prior for Inverse Gamma 

    Returns
    -------
    FLOAT, NxN-dim array
        prior pdf over alpha & beta for Inverse Gamma distribution

    """
    #conjugate prior for alpha
    alpha_prior = (a ** (-alpha - 1)) * (beta[:,np.newaxis] ** (alpha * c)) / (sp.special.gamma(alpha)**b)
    
    #conjugate prior for beta
    beta_prior = (e**d) * (beta[:,np.newaxis]**(d-1)) * np.exp(-e*beta[:,np.newaxis]) / sp.special.gamma(d)
    
    return alpha_prior * beta_prior