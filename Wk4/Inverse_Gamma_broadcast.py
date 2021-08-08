# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:57:30 2021

@author: pyliu
"""
import numpy as np
import scipy as sp

def Inverse_Gamma_broadcast(x, alpha, beta):
    """
    pdf of Inverse Gamma distribution, using broadcasting

    Parameters
    ----------
    x : FLOAT, vector
        Independent variable input
    alpha : FLOAT, scalar
        Distribution parameter 
    beta : FLOAT, scalar
        Distribution parameter 

    Returns
    -------
    FLOAT, vector
        Dependent variable output

    """
    gamma_const = sp.special.gamma(alpha)
    norm_const = (beta[:,np.newaxis]**alpha) / gamma_const
    exponent = -beta[:,np.newaxis] / x
    return norm_const * x ** (-alpha - 1) * np.exp(exponent)
