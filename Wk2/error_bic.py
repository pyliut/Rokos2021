# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:05:39 2021

@author: pyliu
"""

import numpy as np
from Gaussian import *
from Lognormal import *

def error_bic(t_obs, params, model = "gaussian"):
    """
    Return BIC (Bayesian Information Criterion) for a selected model

    Parameters
    ----------
    t_obs : FLOAT, vector
        observed data
    params : FLOAT, vector
        list of parameters for the likelihood distribution
        For Gaussian model:
            [mean_map, var_map]
        For Lognormal model:
            [mean, var, loc]
    model : STR
        The model for the likelihood distribution. The default is "gaussian".

    Returns
    -------
    bic: FLOAT
        BIC criterion

    """
    
    if model == "gaussian":
        mean_map = params[0]
        var_map = params[1]
        k = 2               # no. of params in model
        log_likelihood = np.sum( np.log( Gaussian(t_obs,mean_map,var_map) ) )
    elif model == "lognormal":
        mean = params[0]
        var = params[1]
        loc = params[2]
        k = 3
        log_likelihood = np.sum( np.log( Lognormal(t_obs,mean,var,loc) ) )
    else:
        return "ERROR: Invalid model"
        
    
    bic = k*np.log(len(t_obs)) - 2*log_likelihood
    return bic