# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:20:50 2021

@author: pyliu
"""
import numpy as np
import scipy as sp

def Inverse_Gaussian_broadcast(x, mu, L):
    """
    Inverse Gaussian pdf with broadcasting
    (i.e. all inputs are vectors)

    Parameters
    ----------
    x : FLOAT, vector
        independent variable
    mu : FLOAT, vector
        parameter 1
    L : FLOAT, vector
        parameter 2

    Returns
    -------
    FLOAT, vector
        pdf over independent 

    """
    norm_const = np.sqrt( L[:,np.newaxis]/(2*np.pi) )
    exponent = (-L[:,np.newaxis] / (2* mu**2 * x)) * np.square( x - mu )
    return norm_const * x**(-3/2) * np.exp(exponent)