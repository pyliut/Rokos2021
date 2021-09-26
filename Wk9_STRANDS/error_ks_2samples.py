# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 16:27:34 2021

@author: pyliu
"""
import numpy as np
from integrate_pdf import *
import matplotlib.pyplot as plt

def error_ks_2samples(t_obs1, t_obs2, plot_graph = True):
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
    cdf_hist1 = integrate_pdf(p_hist1)
    cdf_hist1 /= cdf_hist1[-1]              #normalise cdf
    cdf_hist2 = integrate_pdf(p_hist2)
    cdf_hist2 /= cdf_hist2[-1]              #normalise cdf
    
    #3) Calculate max distance
    D = np.max( np.abs(cdf_hist1 - cdf_hist2) )
    
    #4) Plot for visualisation
    if plot_graph == True:
        plt.plot(t_hist1[:-1], cdf_hist1)
        plt.plot(t_hist2[:-1], cdf_hist2)
        plt.legend(["edge1", "edge2"])
        plt.xlabel("Duration (s)")
        plt.ylabel("probability")
        plt.title( "K-S test: D = " + str( np.round(D,5) ) )
    
    return D