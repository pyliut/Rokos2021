# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:02:45 2021

@author: pyliu
"""
import numpy as np

def Gaussian_ml(t):
    """
    Calculate max. likelihood estimators of Gaussian distribution
    
    Parameters
    ----------
    t : FLOAT, Vector
        Observations

    Returns
    -------
    mean_ml : FLOAT, Scalar
        MLE estimate of mean using t
    var_ml : Float, Scalar
        MLE estimate of var using t

    """
    
    N = len(t)
    offset = np.min(t) - 0.01
    t = np.log(t - offset)
    
    mean_ml = (1/N) * np.sum(t)
    var_ml = (1/N) * np.sum(np.square(t - mean_ml))
    
    #convert to standard form
    s = np.sqrt(var_ml)
    loc = offset
    scale = np.exp(mean_ml)
    
    return s,loc,scale