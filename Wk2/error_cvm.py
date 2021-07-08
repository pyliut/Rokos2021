# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 10:58:57 2021

@author: pyliu
"""

import numpy as np
from integrate_pdf import *
import matplotlib.pyplot as plt
from copy import copy

def error_cvm(t_pred, p_pred, t_obs):
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
    p_hist, t_hist = np.histogram(t_obs, density = True, bins = n_bins, range = (range_min, range_max) );     

    #2) turn both observed & predicted pdfs in to cdfs
    pdf_hist = integrate_pdf(p_hist)
    pdf_hist /= pdf_hist[-1]            #normalise cdf
    p_copy = copy(p_pred)               #avoid changing the original data
    pdf_pred = integrate_pdf(p_copy)    
    pdf_pred /= pdf_pred[-1]            #normalise cdf
    
    #3) Calculate max distance
    
    w_squared = np.sum( np.square(pdf_hist - pdf_pred) ) 
    
    #4) Plot for visualisation
    plt.plot(pdf_hist)
    plt.plot(pdf_pred)
    plt.legend(["observed", "predicted"])
    plt.xlabel("samples")
    plt.ylabel("probability (%)")
    plt.title("CVM test: w_squared = " + str(w_squared))
    
    return w_squared