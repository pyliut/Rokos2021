# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:27:34 2021

@author: pyliu
"""
import numpy as np
from integrate_pdf import *
import matplotlib.pyplot as plt

def error_ks_2samples(t_obs1, t_obs2):
    """
    Calculate Kolmogorov-Smirnov Distance of between 2 samples

    Parameters
    ----------
    t_obs1 : FLOAT, vector
        Set 1 of observed durations
    t_obs1 : FLOAT, vector
        Set 2 of observed durations

    Returns
    -------
    D : FLOAT
        K-S statistic
        supremum (Greatest Lower Bound) of distances between the empirical CDF of the observations and CDF of the predicted distribution

    """
    #1) create histogram of observed values
    n_bins = max(len(t_obs1), len(t_obs2))
    range_min = 0
    range_max = np.max( [np.max(t_obs1), np.max(t_obs2)] )
    range_max = 2*(range_max//5)*5 + 5
    p_hist1, t_hist1 = np.histogram(t_obs1, density = True, bins = n_bins, range = (range_min, range_max) );     
    p_hist2, t_hist2 = np.histogram(t_obs2, density = True, bins = n_bins, range = (range_min, range_max) ); 
    
    #2) turn both observed & predicted pdfs in to cdfs
    pdf_hist1 = integrate_pdf(p_hist1)
    pdf_hist1 /= pdf_hist1[-1]              #normalise cdf
    pdf_hist2 = integrate_pdf(p_hist2)
    pdf_hist2 /= pdf_hist2[-1]              #normalise cdf
    
    #3) Calculate max distance
    D = np.abs( np.max(pdf_hist1 - pdf_hist2) )
    
    #4) Plot for visualisation
    plt.plot(pdf_hist1)
    plt.plot(pdf_hist2)
    plt.legend(["Observed", "Predicted"])
    plt.xlabel("No. of observations")
    plt.ylabel("probability (%)")
    plt.title("K-S test: D = " + str(D))
    
    return D