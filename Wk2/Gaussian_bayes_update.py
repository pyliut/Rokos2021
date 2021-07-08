# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 09:33:20 2021

@author: pyliu
"""

import numpy as np
import matplotlib.pyplot as plt

from Gaussian import *

def Gaussian_bayes_update(mean_test, posterior, t_obs, var = 11, threshold = 0.9):
    """
    Updates MAP estimate of mean 
    Uses Gaussian (conjugate) prior, likelihood & posterior

    Parameters
    ----------
    mean_test : FLOAT, vector
        range of means that were tested
        x-axis of plot
    posterior : FLOAT, vector
        posterior probability of each mean in mean_test
        y-axis of plot
    t_obs : pandas Series
        new observations
    var : FLOAT, scalar
        Known variance for Gaussian likelihood. The default is 11.
    threshold: FLOAT, scalar
        Maximum mode of posterior for MAP estimation. The default is 0.9.
    

    Returns
    -------
    mean_test : FLOAT, vector
        range of means that were tested
        x-axis of plot
    posterior : FLOAT, vector
        posterior probability of each mean in mean_test
        y-axis of plot
    mean_map : FLOAT, scalar
        MAP estimate of mean is mode(posterior)

    """
    #1) store initial posterior
    posterior_initial = posterior

    #2) successive updates
    for n in range(1, len(t_obs)):
        #test
        if np.max(posterior) > threshold:
            print("MAP probability above threshold")
            break
        
        t_new = float(t_obs[n:n+1])      #new value
        likelihood = Gaussian(t_new,mean_test,var)
        #Bayes rule
        posterior = posterior*likelihood
        #normalise the distribution
        spacing = mean_test[1]-mean_test[0]
        norm_const = spacing * np.sum(posterior)
        posterior /= norm_const

    #3) Find MAP value
    posterior_max = np.max(posterior)
    posterior_index = np.argmax(posterior)
    mean_map = mean_test[posterior_index]
    
    #4) Plot
    plt.plot(mean_test,posterior)
    plt.plot(mean_test,posterior_initial)
    plt.legend(["posterior", "initial posterior"])
    plt.xlabel("mean")
    plt.ylabel("probability")
    
    #5) calculate error

    return mean_test, posterior, mean_map