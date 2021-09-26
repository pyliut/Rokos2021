# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:58:03 2021

@author: pyliu
"""
import numpy as np

def Gaussian(x,mean,var):
    """
    Calculate Gaussian distribution

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
    return ( 1/ ((2*np.pi*var)**0.5) )*np.exp( (-1/(2*var)) * np.square(x-mean) )