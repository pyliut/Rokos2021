# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:26:51 2021

@author: pyliu
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from Gamma import *

from Inverse_Gaussian import *
from Inverse_Gaussian_broadcast import *
from Inverse_Gaussian_prior import *


def Inverse_Gaussian_bayes(t_obs, a=1, b=1, c=1, d=1, threshold = 0.9):

    
    #error check
    if threshold > 1.0:
        return "ERROR: 0 < threshold < 1 "
    
    #1a) define a range of means to test
    mu_start= 0.1
    mu_stop = 20   #round up to nearest 5 secs
    mu_step = 0.1
    mu_test = np.arange(mu_start,mu_stop,mu_step)
    
    #1b) define a range of variances to test
    L_start= 0.1
    L_stop = 500   #round up to nearest 5 secs
    L_step = 0.1
    L_test = np.arange(L_start,L_stop,L_step)
    
    #2a) create prior
    prior = Inverse_Gaussian_prior(mu_test, L_test, a, b, c, d)
    
    #2b) calculate likelihood from known variance
    t_new = float(t_obs[0:1])
    likelihood = Inverse_Gaussian_broadcast(t_new,mu_test,L_test)
    
    #2c) Bayes rule
    posterior = prior*likelihood
    
    #2d) normalise the distribution
    spacing1 = mu_test[1]-mu_test[0]
    spacing2 = L_test[1]-L_test[0]
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
        likelihood = Inverse_Gaussian_broadcast(t_new,mu_test,L_test)
        #Bayes rule
        posterior = posterior*likelihood
        #normalise the distribution
        spacing1 = mu_test[1]-mu_test[0]
        spacing2 = L_test[1]-L_test[0]
        norm_const = spacing1 *np.sum(posterior)
        posterior /= norm_const
    
    #4) Find MAP value
    p_max = np.amax(posterior)
    ind = np.argwhere(posterior == p_max)
    # ind[0,0] corresponds to index of MAP estimate of var
    # ind[0,1] corresponds to index of MAP estimate of mean  
    mu_map = mu_test[ind[0,1]]
    L_map = L_test[ind[0,0]]
    
    
    #5) Plot
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.contour3D(L_test, mu_test, posterior, 50, cmap=cm.coolwarm)
    
    #ax.set_xlabel('Lambda')
    #ax.set_ylabel('Mu')
    #ax.set_zlabel('prob');
    
    return mu_map,L_map

    
    
    