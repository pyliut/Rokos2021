# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:12:46 2021

@author: pyliu
"""

def update_var(mean, var, n, t_new):
    """
    Sequential update of Gaussian MLE for sample var
    Key weakness: new data gets progressively less powerful in changing the var
    
    Parameters
    ----------
    mean : FLOAT, scalar
        previously calculated mean for n = N - 1 terms
    var : FLOAT, scalar
        previously calculated var for n = N - 1 terms
    n : INT, scalar
        number of observations exluding the current observation
    t_new : FLOAT, scalar
        current observation

    Returns
    -------
    FLOAT, scalar
        updated sample var

    """
    N = n+1
    return ( (N-1)/N ) * var + ( (t_new - mean)**2 )/N