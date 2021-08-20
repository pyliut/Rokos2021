# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 16:44:19 2021

@author: pyliu
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from copy import copy
from integrate_pdf import *

def error_cvm_2samples(t_obs1, t_obs2, plot_graph = True):
    """
    Calculate Cramer-von Mises Statistic 2 samples

    Parameters
    ----------
    t_obs1 : FLOAT, vector
        Set 1 of observed durations
    t_obs1 : FLOAT, vector
        Set 2 of observed durations

    Returns
    -------
    w_squared : FLOAT
        CVM statistic 

    """
    #1) create histogram of observed values
    n_bins = max(len(t_obs1), len(t_obs2))
    range_min = 0
    range_max = np.max( [np.max(t_obs1), np.max(t_obs2)] )
    range_max = 2*(range_max//5)*5 + 5
    bin_width = (range_max - range_min) / n_bins
    p_hist1, t_hist1 = np.histogram(t_obs1, density = True, bins = n_bins, range = (range_min, range_max) );     
    p_hist2, t_hist2 = np.histogram(t_obs2, density = True, bins = n_bins, range = (range_min, range_max) ); 
    
    #2) turn both observed & predicted pdfs in to cdfs
    cdf_hist1 = integrate_pdf(p_hist1)
    cdf_hist1 /= cdf_hist1[-1]              #normalise cdf
    cdf_hist2 = integrate_pdf(p_hist2)
    cdf_hist2 /= cdf_hist2[-1]              #normalise cdf
    
    #3) Calculate max distance
    
    w_squared1 = np.sum( np.square(cdf_hist1 - cdf_hist2) * bin_width*p_hist1 ) * len(t_obs1)
    w_squared2 = np.sum( np.square(cdf_hist1 - cdf_hist2) * bin_width*p_hist2 ) * len(t_obs2)
    
    #4) Plot for visualisation
    if plot_graph == True:
        plt.plot(t_hist1[:-1], cdf_hist1)
        plt.plot(t_hist2[:-1], cdf_hist2)
        plt.legend(["edge1", "edge2"])
        plt.xlabel("Duration (s)")
        plt.ylabel("probability")
        plt.title("CVM test: w_squared1 = " + str( np.round(w_squared1, 5) ))
        
    return w_squared1, w_squared2