# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:10:03 2021

@author: pyliu
"""
import numpy as np
def update_mean(mean, n, t_new):
    """
    Sequential update of Gaussian MLE for mean (Bishop 2.3.5)
    Key weakness: new data gets progressively less powerful in changing the mean
    
    Parameters
    ----------
    mean : FLOAT, scalar
        previously calculated mean for n = N - 1 terms
    n : INT, scalar
        number of observations exluding the current observation
    t_new : FLOAT, scalar
        current observation

    Returns
    -------
    FLOAT, scalar
        updated mean

    """
    
    
    N = n+1
    return mean + (1/N)*(t_new - mean)

