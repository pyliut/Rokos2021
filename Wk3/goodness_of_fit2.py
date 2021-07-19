# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:08:44 2021

@author: pyliu
"""

import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


from update_mean import *

def goodness_of_fit2(t_obs, n_iter = 100, model = stats.norm):
    """
    Finds KS statistic & p_value, MAE, MSE of t_obs against a proposed distribution
    Does NOT plot the fitted distribution against the observed data (t_obs)

    Parameters
    ----------
    t_obs : FLOAT, vector
        observations
    n_iter : INT, scalar
        No. of estimations of each KS/MAE/MSE statistic. The default is 100.
    model : STR
        The scipy.stats model that is the proposed distribution. The default is stats.norm.

    Returns
    -------
    D_mean : FLOAT, scalar
        KS statistic
    p_mean : FLOAT, scalar
        KS p-value
    mae_mean : FLOAT, scalar
        MAE estimate
    mse_mean : FLOAT, scalar
        MSE estimate

    """
    #1) Fit model
    params = model.fit(t_obs)
    if len(params) == 1:
        param0 = params[0]
    elif len(params) == 2:
        param0 = params[0]
        param1 = params[1]
    elif len(params) == 3:
        param0 = params[0]
        param1 = params[1] 
        param2 = params[2] 
    elif len(params) == 4:
        param0 = params[0]
        param1 = params[1] 
        param2 = params[2] 
        param3 = params[3] 
    
    #2a) initialise test metrics
    #K-S Test
    D_mean = 0
    p_mean = 0
    #MAE/MSE
    mae_mean = 0
    mse_mean = 0
    
    #3) incremental update to means
    for i in range(n_iter):
        #new set of random variables
        if len(params) == 1:
            t_pred = model.rvs(param0, size = len(t_obs))
        elif len(params) == 2:
            t_pred = model.rvs(param0, param1, size = len(t_obs))
        elif len(params) == 3:
            t_pred = model.rvs(param0, param1, param2, size = len(t_obs))
        elif len(params) == 4:
            t_pred = model.rvs(param0, param1, param2, param3, size = len(t_obs))
        
        #Update KS test
        D,p = stats.ks_2samp(t_obs, t_pred)
        D_mean = update_mean(D_mean, i, D)
        p_mean = update_mean(p_mean, i, p)
        
        #Update MAE/MSE
        mae = ( 1/len(t_obs) ) * np.sum( np.abs(t_obs - t_pred) )
        mse = ( 1/len(t_obs) ) * np.sum( np.square(t_obs - t_pred) )
        mae_mean = update_mean(mae_mean, i, mae)
        mse_mean = update_mean(mse_mean, i, mse)
    
    
    return D_mean, p_mean, mae_mean, mse_mean