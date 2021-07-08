# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:37:13 2021

@author: pyliu
"""

import numpy as np


def error_mse(t_pred, p_pred, t_obs, bin_width = 20):
    """
    Calculate Mean Square Error (MSE) of a predicted distribution vs a histogram of observations

    Parameters
    ----------
    t_pred : FLOAT, vector
        predicted duration
        x-axis of predicted distribution
    p_pred : FLOAT, vector
        predicted probability
        y-axis of predicted distribution
    t_obs : FLOAT, vector
        observed durations
    bin_width : INT
        width of histogram bins for observations. The default is 20.

    Returns
    -------
    error_mse : INT
        Mean Square Error (MSE)

    """
    #1) create histogram of observed values
    n_bins = len(t_pred)//bin_width
    range_min = np.min(t_pred)
    range_max = np.max(t_pred)
    p_hist, t_hist = np.histogram(t_obs, density = True, bins = n_bins, range = (range_min, range_max) );     

    #2) Calc. mean absolute error (MAE)
    error_mse = 0
    for i in range(n_bins):
        error_mse += np.sum( np.square( p_hist[i] - p_pred[i*bin_width : (i+1)*bin_width] ) )
    error_mse /= len(t_pred)
    
    return error_mse