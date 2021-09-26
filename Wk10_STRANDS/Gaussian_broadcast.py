# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 13:53:19 2021

@author: pyliu
"""

import numpy as np

def Gaussian_broadcast(x,mean,var):
    """
    Calculate Gaussian distribution using broadcasting
    Take an N-dim input mean and N-dim input var
    Returns NxN-dim probability matrix

    Parameters
    ----------
    x : FLOAT/pandas Series, Scalar or Vector
        Independent variable e.g. time
    mean : FLOAT, series
        Parameter of Gaussian distribution
    var : FLOAT, series
        Parameter of Gaussian distribution

    Returns
    -------
    FLOAT, Scalar or Vector
        Calculation of Gaussian distribution probability

    """
    return ( 1/ ((2*np.pi*var[:,np.newaxis])**0.5) ) * np.exp( (-1/(2*var[:,np.newaxis])) * np.square(x-mean) )