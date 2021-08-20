# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:07:59 2021

@author: pyliu
"""
import numpy as np
from integrate_pdf import *
import matplotlib.pyplot as plt
from copy import copy

def error_ks(t_pred, p_pred, t_obs, plot_graph = True):
    """
    Calculate Kolmogorov-Smirnov Distance of between observations and predicted distribution

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

    Returns
    -------
    D : FLOAT
        K-S statistic
        supremum (Greatest Lower Bound) of distances between the empirical CDF of the observations and CDF of the predicted distribution

    """
    #1) create histogram of observed values
    n_bins = len(t_pred)
    range_min = np.min(t_pred)
    range_max = np.max(t_pred)
    p_hist, t_hist = np.histogram(t_obs, density = True, bins = n_bins, range = (range_min, range_max) );     

    #2) turn both observed & predicted pdfs in to cdfs
    cdf_hist = integrate_pdf(p_hist)
    cdf_hist /= cdf_hist[-1]            #normalise cdf
    p_copy = copy(p_pred)               #avoid changing the original data
    cdf_pred = integrate_pdf(p_copy)    
    cdf_pred /= cdf_pred[-1]            #normalise cdf
    
    #3) Calculate max distance
    D = np.max( np.abs(cdf_pred - cdf_hist) )
    
    #4) Plot for visualisation
    if plot_graph == True:
        plt.plot(t_hist[:-1], cdf_hist)
        plt.plot(t_pred, cdf_pred)
        plt.legend(["Observed", "Predicted"])
        plt.xlabel("Duration (s)")
        plt.ylabel("probability")
        plt.title( "K-S test: D = " + str( np.round(D,5) ) )
    
    return D
    