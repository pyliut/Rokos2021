# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:22:13 2021

@author: pyliu
"""
import numpy as np
import scipy as sp
from Gamma import *

def Inverse_Gaussian_prior(mu,L,a,b,c,d):
    """
    Calculates pdf of the prior for Inverse Gaussian

    Parameters
    ----------
    mu : FLOAT, N-dim vector
        parameter of Inverse Gaussian
    L : FLOAT, N-dim vector
        parameter of Inverse Gaussian
    a : FLOAT, scalar
        hyperparameter (of prior) for Inverse Gaussian
    b : FLOAT, scalar
        hyperparameter (of prior) for Inverse Gaussian
    c : FLOAT, scalar
        hyperparameter (of prior) for Inverse Gaussian
    d : FLOAT, scalar
        hyperparameter (of prior) for Inverse Gaussian

    Returns
    -------
    FLOAT, NxN-dim array
        pdf of the prior for Inverse Gaussian

    """
    gamma_mu = Gamma(mu,a,b)
    gamma_L = Gamma(L,c,d)
    return gamma_mu * gamma_L[:,np.newaxis]