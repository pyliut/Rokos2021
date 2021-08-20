# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:44:14 2021

@author: pyliu
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from copy import copy
from integrate_pdf import *

def error_cvm(t_pred, p_pred, t_obs, plot_graph = True):
    """
    Calculate Cramer-von Mises Statistic between observations and predicted distribution

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
    w_squared : FLOAT
        CVM statistic 

    """
    #1) create histogram of observed values
    n_bins = len(t_pred)
    range_min = np.min(t_pred)
    range_max = np.max(t_pred)
    bin_width = (range_max - range_min) / n_bins
    p_hist, t_hist = np.histogram(t_obs, density = True, bins = n_bins, range = (range_min, range_max) );     

    #2) turn both observed & predicted pdfs in to cdfs
    cdf_hist = integrate_pdf(p_hist)
    cdf_hist /= cdf_hist[-1]            #normalise cdf
    p_copy = copy(p_pred)               #avoid changing the original data
    cdf_pred = integrate_pdf(p_copy)    
    cdf_pred /= cdf_pred[-1]            #normalise cdf
    
    #3) Calculate max distance
    
    w_squared = np.sum( np.square(cdf_hist - cdf_pred) * bin_width*p_pred ) * len(t_obs)
    
    #4) Plot for visualisation
    if plot_graph == True:
        plt.plot(t_hist[:-1], cdf_hist)
        plt.plot(t_pred, cdf_pred)
        plt.legend(["observed", "predicted"])
        plt.xlabel("duration (s)")
        plt.ylabel("probability")
        plt.title("CVM test: w_squared = " + str( np.round(w_squared, 5) ))
        
    return w_squared
