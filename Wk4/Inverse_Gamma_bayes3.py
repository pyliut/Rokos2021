# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 13:03:41 2021

@author: pyliu
"""
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from matplotlib import cm

from Inverse_Gamma_broadcast import *
from Inverse_Gamma_prior import *

def Inverse_Gamma_bayes3(t_obs, a=1, b=1, c=1, d=1, e=1, threshold = 0.9):
    """
    

    Parameters
    ----------
    t_obs : FLOAT, vector
        observed durations
    a : FLOAT, scalar
        parameter of prior for Inverse Gamma. The default is 1.
    b : FLOAT, scalar
        parameter of prior for Inverse Gamma. The default is 1.
    c : FLOAT, scalar
        parameter of prior for Inverse Gamma. The default is 1.
    d : FLOAT, scalar
        parameter of prior for Inverse Gamma. The default is 1.
    e : FLOAT, scalar
        parameter of prior for Inverse Gamma. The default is 1.
    threshold : FLOAT, scalar
        max probability of posterior. The default is 0.9.

    Returns
    -------
    alpha_map: FLOAT, scalar
        MAP estimate of Inverse Gamma parameter
    beta_map: FLOAT, scalar
        MAP estimate of Inverse Gamma parameter
    """
    
    #error check
    if threshold > 1.0:
        return "ERROR: 0 < threshold < 1 "
    
    #1a) define a range of means to test
    alpha_start= 0.1
    alpha_stop = 20  #round up to nearest 5 secs
    alpha_step = 0.1
    alpha_test = np.arange(alpha_start,alpha_stop,alpha_step)
    
    #1b) define a range of variances to test
    beta_start= 0.1
    beta_stop = 500   #round up to nearest 5 secs
    beta_step = 0.1
    beta_test = np.arange(beta_start,beta_stop,beta_step)
    
    #2a) create prior
    prior = Inverse_Gamma_prior(alpha_test, beta_test, a, b, c, d, e)
    
    #2b) calculate likelihood from known variance
    t_new = float(t_obs[0:1])
    likelihood = Inverse_Gamma_broadcast(t_new,alpha_test,beta_test)
    
    #2c) Bayes rule
    posterior = prior*likelihood
    
    #2d) normalise the distribution
    spacing1 = alpha_test[1]-alpha_test[0]
    spacing2 = beta_test[1]-beta_test[0]
    norm_const = spacing1*np.sum(posterior)
    posterior /= norm_const
    
    #2e) store initial posterior & likelihood
    posterior_initial = posterior
    likelihood_initial = likelihood 
    
    #3) successive updates
    for n in range(1, len(t_obs)):
        #test
        if np.max(posterior) > threshold:
            print(n, "MAP probability above threshold")
            break
    
        t_new = float(t_obs[n:n+1])      #new value
        likelihood = Inverse_Gamma_broadcast(t_new,alpha_test,beta_test)
        #Bayes rule
        posterior = posterior*likelihood
        #normalise the distribution
        spacing1 = alpha_test[1]-alpha_test[0]
        spacing2 = beta_test[1]-beta_test[0]
        norm_const = spacing1 *np.sum(posterior)
        posterior /= norm_const
    
    #4) Find MAP value
    p_max = np.amax(posterior)
    ind = np.argwhere(posterior == p_max)
    # ind[0,0] corresponds to index of MAP estimate of var
    # ind[0,1] corresponds to index of MAP estimate of mean  
    alpha_map = alpha_test[ind[0,1]]
    beta_map = beta_test[ind[0,0]]
    
    
    #5) Plot
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.contour3D(alpha_test, beta_test, posterior, 50, cmap=cm.coolwarm)
    
    #ax.set_xlabel('Alpha')
    #ax.set_ylabel('Beta')
    #ax.set_zlabel('prob');
    
    return alpha_map, beta_map
