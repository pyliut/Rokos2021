# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:17:22 2021

@author: pyliu
"""
import numpy as np
import scipy as sp

def Inverse_Gaussian(x, mu, L):
    """
    Inverse Gaussian pdf

    Parameters
    ----------
    x : FLOAT, vector
        independent variable
    mu : FLOAT, scalar
        parameter 1
    L : FLOAT, scalar
        parameter 2

    Returns
    -------
    FLOAT, vector
        pdf over independent 

    """
    norm_const = np.sqrt( L/(2*np.pi) )
    exponent = (-L / (2* mu**2 * x)) * np.square( x - mu )
    return norm_const * x**(-3/2) * np.exp(exponent)