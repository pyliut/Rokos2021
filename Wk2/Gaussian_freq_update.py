# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:55:51 2021

@author: pyliu
"""
import numpy as np
import matplotlib.pyplot as plt
from update_mean import *
from update_var import *
from Gaussian import *

def Gaussian_freq_update(mean_ml,var_ml,n,t_new):
    """
    Updates the frequentist MLE estimate of mean & variance of Gaussian distribution
    Also plots the predictions before and after the update

    Parameters
    ----------
    mean_ml : FLOAT, scalar
        Old MLE of mean
    var_ml : FLOAT, scalar
        Old MLE of variance
    n : INT, scalar
        Number of terms prior to update
    t_new : pandas Series, vector (can also be scalar)
        New observations

    Returns
    -------
    t : FLOAT, vector
        duration (s)
        data in x-axis of plot
    p : FLOAT, vector
        probability
        data in y-axis of line plot
    mean_ml : FLOAT, scalar
        Updated MLE of mean
    var_ml : FLOAT, scalar
        Updated MLE of variance

    """
    
    #keep for comparison
    n_old = n           
    mean_old = mean_ml
    var_old = var_ml
    
    #1) perform sequential updates
    for i in range(len(t_new)):
        t_new_float= float(t_new[i:i+1])
        mean_ml = update_mean(mean_ml, n, t_new_float)
        var_ml = update_var(mean_ml, var_ml, n, t_new_float)
        n += 1
    print(mean_ml)
    #2) Calculate old and new predictive distributions
    t_start= 0
    t_stop = ( (3*mean_ml) //5)*5 + 5    #round up to nearest 5 secs
    t_step = 0.01
    t = np.arange(t_start,t_stop,t_step)
    
    #predict duration distribution
    p_old = Gaussian(t,mean_old,var_old)
    p = Gaussian(t,mean_ml,var_ml)
    
    #3) Plot
    plt.plot(t,p)
    plt.plot(t,p_old)
    plt.legend(["Updated","Old"])
    plt.xlabel("operation_time (s)")
    plt.ylabel("probability")
    
    #4) Error
    
    return t, p, mean_ml, var_ml