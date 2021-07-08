# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 11:44:21 2021

@author: pyliu
"""

import numpy as np
from integrate_pdf import *
import matplotlib.pyplot as plt

def error_cvm_2samples(t_obs1, t_obs2, n_bins = 1000, t_max = 40):
    """
    Calculate Cramer-von Mises Statistic of between 2 samples
    Note: CVM statistics should be calculated with same n_bins & t_max in order to be directly compared

    Parameters
    ----------
    t_obs1 : FLOAT, vector
        Set 1 of observed durations
    t_obs1 : FLOAT, vector
        Set 2 of observed durations
    n_bins: INT
        no. of intervals that the t-axis is divided into. The default is 1000.
    t_max: INT
        max value on t-axis. The default is 40.
    

    Returns
    -------
    D : FLOAT
        CVM statistic
        
    """
    #1) create histogram of observed values
    p_hist1, t_hist1 = np.histogram(t_obs1, density = True, bins = n_bins, range = (0, t_max) );     
    p_hist2, t_hist2 = np.histogram(t_obs2, density = True, bins = n_bins, range = (0, t_max) ); 
    
    #2) turn both observed & predicted pdfs in to cdfs
    pdf_hist1 = integrate_pdf(p_hist1)
    pdf_hist1 /= pdf_hist1[-1]              #normalise cdf
    pdf_hist2 = integrate_pdf(p_hist2)
    pdf_hist2 /= pdf_hist2[-1]              #normalise cdf
    
    #3) Calculate max distance
    w_squared = np.sum( np.square(pdf_hist1 - pdf_hist2) ) 
    
    #4) Plot for visualisation
    plt.plot(pdf_hist1)
    plt.plot(pdf_hist2)
    plt.legend(["Observed", "Predicted"])
    plt.xlabel("No. of observations")
    plt.ylabel("probability")
    plt.title("CVM test: w_squared = " + str(w_squared))
    
    return w_squared