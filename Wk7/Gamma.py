# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:41:13 2021

@author: pyliu
"""
import math
import numpy as np


def Gamma(L,a,b):
    """
    Calculate Gamma distribution

    Parameters
    ----------
    L : FLOAT/pandas Series, scalar or vector
        precision - inverse of variance
    a : FLOAT, scalar
        parameter of Gamma distribution
    b : FLOAT, scalar
        parameter of Gamma distribution

    Returns
    -------
    FLOAT/pandas Series
        Calculation of Gamma distribution probability

    """
    
    
    gamma_a = math.gamma(a)
    return (1/gamma_a) * (b**a) * np.multiply( (L ** (a-1)), np.exp(-b*L) )