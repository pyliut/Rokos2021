# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 12:58:22 2021

@author: pyliu
"""
import numpy as np
from Gaussian import *

def Gaussian_log_likelihood(t_obs, mean_map, var_map):
    """
    Returns log_likelihood of Gaussian model

    Parameters
    ----------
    t_obs : FLOAT, vector
        observed data
    mean_map : FLOAT, scalar
        MAP estimate of mean of Gaussian
    var_map : FLOAT, scalar
        MAP estimate of variance of Gaussian

    Returns
    -------
    log_likelihood : FLOAT, scalar
        log_likelihood of Gaussian model

    """
    log_likelihood = np.sum( np.log( Gaussian(t_obs,mean_map,var_map) ) )
    return log_likelihood