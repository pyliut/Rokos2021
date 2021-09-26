# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:25:31 2021

@author: pyliu
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from Gaussian_broadcast import *
from Gamma import *
from Normal_Gamma import *

def Normal_Gamma_bayes_update(mean_test, var_test, posterior, t_obs, threshold = 0.9, plot_graph = False):
    """
    Bayesian updates for additional len(t_obs) observations
    Assumes a unknown variance and unknown mean
    Uses Normal-Gamma (conjugate) prior & posterior
    With Normal likelihood

    Parameters
    ----------
    mean_test : FLOAT, vector
        range of means that were tested
        x-axis of plot
    var_test : FLOAT, vector
        range of variances that were tested
        y-axis of plot
    posterior : FLOAT, vector
        initial posterior probability of each mean in mean_test & var in var_test
        z-axis of plot
    t_obs : pandas Series
        new observations
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
        updated posterior probability of each mean in mean_test & var in var_test
        z-axis of plot
    mean_map : FLOAT, scalar
        updated MAP estimate of mean is mode(posterior) wrt mean_test
    var_map : FLOAT, scalar
        updated MAP estimate of mean is mode(posterior) wrt var_test


    """
    
    #error check
    if threshold > 1.0:
        return "ERROR: 0 < threshold < 1 "
    
    #1) store initial posterior
    #posterior_initial = posterior
    
    #2) successive updates
    for n in range(len(t_obs)):
        #test
        #if np.max(posterior) > threshold:
            #print("MAP probability above threshold")
            #break
    
        t_new = float(t_obs[n:n+1])      #new value
        likelihood = Gaussian_broadcast(t_new,mean_test,var_test)
        #Bayes rule
        posterior = posterior*likelihood
        #normalise the distribution
        spacing = mean_test[1]-mean_test[0]
        norm_const = spacing * np.sum(posterior)
        posterior /= norm_const

    #3) Find MAP value
    p_max = np.amax(posterior)
    ind = np.argwhere(posterior == p_max)
    # ind[0,0] corresponds to index of MAP estimate of var
    # ind[0,1] corresponds to index of MAP estimate of mean  
    mean_map = mean_test[ind[0,1]]
    var_map = var_test[ind[0,0]]
    
    #4) Plot
    if plot_graph == True:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(var_test, mean_test, posterior, 50, cmap=cm.coolwarm)
        ax.set_xlabel('var')
        ax.set_ylabel('mean')
        ax.set_zlabel('prob');
    
    #5) calculate error
    return mean_test, var_test, posterior, mean_map, var_map