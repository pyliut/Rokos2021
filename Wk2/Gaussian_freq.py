# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:18:55 2021

@author: pyliu
"""
import numpy as np
import matplotlib.pyplot as plt
from Gaussian_ml import *
from Gaussian import *

def Gaussian_freq(t_obs):
    """
    Frequentist MLE estimate of mean & variance of Gaussian distribution
    Also plots the predicted and actual data

    Parameters
    ----------
    t_obs : pandas Series, vector (can also be scalar)
        observed data

    Returns
    -------
    t : FLOAT, vector
        duration (s)
        data in x-axis of plot
    p : FLOAT, vector
        probability
        data in y-axis of line plot
    mean_ml : FLOAT, scalar
        MLE of mean
    var_ml : FLOAT, scalar
        MLE of variance

    """
    
    #1) find mean & variance of observations
    mean_ml, var_ml = Gaussian_ml(t_obs)
    
    #2) calculate predictive distribution
    t_start= 0
    t_stop = (np.max(t_obs)//5)*5 + 5    #round up to nearest 5 secs
    t_step = 0.01
    t = np.arange(t_start,t_stop,t_step)
    #predict duration distribution
    p = Gaussian(t,mean_ml,var_ml)
    
    #3) Plot
    plt.plot(t,p)
    plt.hist(t_obs, density = True, bins = 50)
    plt.legend(["predicted","actual"])
    plt.xlabel("operation_time (s)")
    plt.ylabel("probability")
    
    #4) Error
    
    return t, p, mean_ml, var_ml
    
    
    
    