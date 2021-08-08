# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 13:38:48 2021

@author: pyliu
"""
import numpy as np

def Lognormal(x,mean,var):
    """
    Calculate Lognormal distribution

    Parameters
    ----------
    x : FLOAT/pandas Series, Scalar or Vector
        Independent variable e.g. time
    mean : FLOAT
        Parameter of Gaussian distribution
    var : FLOAT
        Parameter of Gaussian distribution

    Returns
    -------
    FLOAT, Scalar or Vector
        Calculation of Gaussian distribution probability

    """
    x = np.array(x)
    n_total = len(x)
    n_pos = len(x[x>0])
    n_neg = n_total - n_pos
    neg_array = np.zeros(n_neg)
    x = x[x>0]
    
    norm_const = 1/ np.multiply( x, ( (2*np.pi*var)**0.5 ) )
    exponent = (-1/(2*var)) * np.square( np.log(x)-mean )
    result = np.multiply( norm_const, np.exp(exponent) )
    
    result = [*neg_array, *result]
    return  np.array(result)