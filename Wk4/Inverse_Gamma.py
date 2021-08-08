# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:41:41 2021

@author: pyliu
"""
import numpy as np
import scipy as sp

def Inverse_Gamma(x, alpha, beta):
    """
    pdf of Inverse Gamma distribution

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
    norm_const = (beta**alpha) / gamma_const
    exponent = -beta / x
    return norm_const * x ** (-alpha - 1) * np.exp(exponent)