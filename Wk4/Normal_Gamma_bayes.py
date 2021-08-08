# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:50:45 2021

@author: pyliu
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from Gaussian_broadcast import *
from Gamma import *
from Normal_Gamma import *

def Normal_Gamma_bayes(t_obs, mu_0 = 10, beta = 5, a = 1, b = 2, threshold = 0.9, plot_graph = True):
    """
    Bayesian updates for 1st len(t_obs) observations
    Assumes a unknown variance and unknown mean
    Uses Normal-Gamma (conjugate) prior & posterior
    With Normal likelihood

    Parameters
    ----------
    t_obs : pandas Series
        observations
    mu_0 : FLOAT, scalar
        Parameter for prior. The default is 10.
    beta : FLOAT, scalar
        Parameter for prior. The default is 5.
    a : FLOAT, scalar
        Parameter for prior. The default is 1.
    b : FLOAT, scalar
        Parameter for prior. The default is 2.
    threshold: FLOAT, scalar
        Maximum mode of posterior for MAP estimation. The default is 0.9.
    
    Returns
    -------
    mean_test : FLOAT, vector
        range of means that were tested
        x-axis of plot
    var_test : FLOAT, vector
        range of variances that were tested
        y-axis of plot
    posterior : FLOAT, vector
        posterior probability of each mean in mean_test & var in var_test
        z-axis of plot
    mean_map : FLOAT, scalar
        MAP estimate of mean is mode(posterior) wrt mean_test
    var_map : FLOAT, scalar
        MAP estimate of mean is mode(posterior) wrt var_test

    """
    #error check
    if threshold > 1.0:
        return "ERROR: 0 < threshold < 1 "
    
    #1a) define a range of means to test
    mean_start= 0.001
    mean_stop = np.max(t_obs)   #round up to nearest 5 secs
    mean_step = 0.001
    mean_test = np.arange(mean_start,mean_stop,mean_step)
    
    #1b) define a range of variances to test
    var_start= 0.001
    var_stop = np.max(t_obs)   #round up to nearest 5 secs
    var_step = 0.001
    var_test = np.arange(var_start,var_stop,var_step)
    
    #2a) create Normal-Gamma prior
    prior = Normal_Gamma(mean_test, var_test, mu_0, beta, a, b)
    
    #2b) calculate likelihood from known variance
    t_new = float(t_obs[0:1])
    likelihood = Gaussian_broadcast(t_new,mean_test,var_test)
    
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
        likelihood = Gaussian_broadcast(t_new,mean_test,var_test)
        #Bayes rule
        posterior = posterior*likelihood
        #normalise the distribution
        spacing = mean_test[1]-mean_test[0]
        norm_const = spacing * np.sum(posterior)
        posterior /= norm_const

    #4) Find MAP value
    p_max = np.amax(posterior)
    ind = np.argwhere(posterior == p_max)
    # ind[0,0] corresponds to index of MAP estimate of var
    # ind[0,1] corresponds to index of MAP estimate of mean  
    mean_map = mean_test[ind[0,1]]
    var_map = var_test[ind[0,0]]
    
    #5) Plot
    if plot_graph == True:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(var_test, mean_test, posterior, 50, cmap=cm.coolwarm)
        
        ax.set_xlabel('var')
        ax.set_ylabel('mean')
        ax.set_zlabel('prob');
    
    #6) calculate error

    return mean_test, var_test, posterior, mean_map, var_map