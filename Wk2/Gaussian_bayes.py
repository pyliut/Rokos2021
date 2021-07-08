# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:30:08 2021

@author: pyliu
"""
import numpy as np
import matplotlib.pyplot as plt

from Gaussian import *

def Gaussian_bayes(t_obs, mean_0 = 10, var_0 = 10, var = 11, threshold = 0.9):
    """
    Bayesian updates for 1st len(t_obs) observations
    Assumes a known variance and unknown mean
    Uses Gaussian (conjugate) prior, likelhood & posterior

    Parameters
    ----------
    t_obs : pandas Series
        observations
    mean_0 : FLOAT, scalar
        Gaussian parameter for prior. The default is 10.
    var_0 : FLOAT, scalar
        Gaussian parameter for prior. The default is 10.
    var : FLOAT, scalar
        Known variance for Gaussian likelihood. The default is 11.
    threshold: FLOAT, scalar
        Maximum mode of posterior for MAP estimation
    
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
    if threshold > 1.0:
        return "ERROR: 0 < threshold < 1 "
    #1) define a range of means to test
    mean_start= 0
    mean_stop = np.max( [ 3*(np.max(t_obs)//5)*5 + 5, 40, 3*(mean_0//5)*5] )   #round up to nearest 5 secs
    mean_step = 0.001
    mean_test = np.arange(mean_start,mean_stop,mean_step)
    
    #2a) create Gaussian prior
    prior = Gaussian(mean_test,mean_0,var_0)
    
    #2b) calculate likelihood from known variance
    t_new = float(t_obs[0:1])
    likelihood = Gaussian(t_new,mean_test,var)
    
    #2c) Bayes rule
    posterior = prior*likelihood
    
    #2d) normalise the distribution
    spacing = mean_test[1]-mean_test[0]
    norm_const = spacing * np.sum(posterior)
    posterior /= norm_const
    
    #2e) store initial posterior & likelihood
    posterior_initial = posterior
    likelihood_initial = likelihood 
    
    #3) successive updates
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

    #4) Find MAP value
    posterior_max = np.max(posterior)
    posterior_index = np.argmax(posterior)
    mean_map = mean_test[posterior_index]
    
    #5) Plot
    plt.plot(mean_test,posterior)
    plt.plot(mean_test,posterior_initial)
    plt.plot(mean_test,likelihood_initial)
    plt.plot(mean_test,prior)
    plt.legend(["posterior", "initial posterior", "initial likelihood","prior"])
    plt.xlabel("mean")
    plt.ylabel("probability")
    
    #6) calculate error

    return mean_test, posterior, mean_map
    
    
    