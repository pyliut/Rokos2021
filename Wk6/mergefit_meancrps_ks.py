# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 12:25:49 2021

@author: pyliu
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import math
import scipy as sp
import time

from select_data_edge import *
from Lognormal import *
from Normal_Gamma_bayes import *
from integrate_pdf import *
from error_crps import *
from underscore_prefix import *
from underscore_suffix import *


def mergefit_meancrps_ks(df, edge_list, n_samples = 10, n_std = 2):
    """
    Performs mergefit:
    Fits a lognormal distribution based on n_samples datapoints from each edge in edge_list
    CRPS is calculated as the mean excluding outliers further than n_std standard deviations away from the mean

    Parameters
    ----------
    df : pandas dataframe
        columns = ["origin", "target", "edge_id", "operation_time"]
    edge_list : STR, vector
        list of "edge_id" in a particular cluster
    n_samples : INT, scalar
        number of datapoints taken from each edge in mergefit. The default is 10.

    Returns
    -------
    mergefit_stats : pandas dataframe
        columns = ["edge_id", "n_samples","crps", "ks", "crps_orig", "ks_orig","crps_diff", "ks_diff"]
        crps & ks are statistics for mergefit wrt unseen data for each edge
        crps_orig & ks_orig are statistics for original distribution

    """
    tic = time.time()
    #1) take n_samples datapoints from each edge
    t_op = []
    indices = []
    for edge in edge_list:
        subset = select_data_edge(df, edge)
        #independent variable to plot over
        t = list(subset["operation_time"])
        max_ind = min(n_samples, len(t)//2)
        t = t[0:max_ind]
        indices.append(max_ind)
        t_op = [*t_op, *t]
    
    #2) fit model
    #use offset and take log of data
    offset= np.min(t_op) - 0.01
    t_log = np.log(t_op - offset)
    
    #n_terms = 1 if you want just the initial estimates
    n_terms = len(t_log)    
    t_obs = t_log[0:n_terms]
    
    #set parameters
    mu_0 = 1
    beta = 0.1
    a = 1
    b = 1
    
    #Bayesian MAP estimate of mean & variance of Gaussian distribution
    mean_test, var_test, posterior, mean_map, var_map = Normal_Gamma_bayes(t_obs, mu_0, beta, a, b, plot_graph = False)
    print("MAP mean:", mean_map)
    print("MAP var:", var_map)
    print("Offset:", offset)

    #3) Initialise df to store CRPS
    mergefit_stats = pd.DataFrame(columns = ["edge_id", "n_samples","crps","ks", "crps_orig", "ks_orig", "crps_diff", "ks_diff"])
    mergefit_stats["edge_id"] = edge_list
    n_plots = len(edge_list)
    n_cols = 5
    n_rows = int(np.ceil(n_plots / n_cols))
    
    #4) initialise subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15*n_cols,10*n_rows))
    plt.xlabel("Duration (s)")
    plt.ylabel("Probability")
      
    #5) Calculate CRPS & KS
    for i in range(len(edge_list)):
        edge = edge_list[i]
        subset = select_data_edge(df, edge)
        mergefit_stats["n_samples"][i] = len(subset)
        #independent variable to plot over
        t_op_orig = subset["operation_time"]
        
    
        precision = 2
        t_start= 10**(-precision)
        t_stop = ( (np.max(t_op) + offset) //5)*5 + 5    #round up to nearest 5 secs
        t_step = 10**(-precision)
        t_test = np.arange(t_start,t_stop,t_step)
    
        #Duration distribution using MAP parameters from Bayesian method
        p_bayes = Lognormal(t_test-offset,mean_map,var_map)
        
        #6) fit original edge
        #use offset and take log of data
        offset_orig= np.min(t_op_orig) - 0.01
        t_log_orig = np.log(t_op_orig - offset_orig)
        
        #n_terms = 1 if you want just the initial estimates
        n_terms_orig = len(t_log_orig)    
        t_obs_orig = t_log_orig[0:n_terms_orig]
        
        #set parameters
        mu_0 = 1
        beta = 0.1
        a = 1
        b = 1
        
        #Bayesian MAP estimate of mean & variance of Gaussian distribution
        mean_test_orig, var_test_orig, posterior_orig, mean_map_orig, var_map_orig = Normal_Gamma_bayes(t_obs_orig, mu_0, beta, a, b, plot_graph = False)
        
        #Duration distribution using MAP parameters from Bayesian method
        p_bayes_orig = Lognormal(t_test-offset_orig,mean_map_orig,var_map_orig)

        #7) stats
        #crps score
        cdf_bayes = integrate_pdf(p_bayes, spacing = t_test[1] - t_test[0])
        crps = error_crps(np.array(t_op_orig[indices[i]:]),cdf_bayes,t_test, method = "rectangle")
        crps_mean = crps["crps"].mean()  
        crps_std = crps["crps"].std() 
        crps_filtered = [e for e in crps["crps"] if (crps_mean - n_std * crps_std < e < crps_mean + n_std * crps_std)]
        mergefit_stats["crps"][i] = np.mean(crps_filtered)   
        
        cdf_bayes_orig = integrate_pdf(p_bayes_orig, spacing = t_test[1] - t_test[0])
        crps_orig = error_crps(np.array(t_op_orig[indices[i]:]),cdf_bayes_orig,t_test, method = "rectangle")
        crps_mean = crps_orig["crps"].mean()  
        crps_std = crps_orig["crps"].std() 
        crps_filtered = [e for e in crps_orig["crps"] if (crps_mean - n_std * crps_std < e < crps_mean + n_std * crps_std)]
        mergefit_stats["crps_orig"][i] = np.mean(crps_filtered)  
        
        #ks score
        mergefit_stats["ks"][i] = sp.stats.kstest(t_op_orig[indices[i]:], lambda k: sp.stats.lognorm.cdf(k, s = np.sqrt(var_map), loc = offset, scale = np.exp(mean_map)))
        mergefit_stats["ks_orig"][i] = sp.stats.kstest(t_op_orig[indices[i]:], lambda k: sp.stats.lognorm.cdf(k, s = np.sqrt(var_map_orig), loc = offset_orig, scale = np.exp(mean_map_orig)))
        
        #calculate differences
        mergefit_stats["crps_diff"][i] = mergefit_stats["crps"][i] - mergefit_stats["crps_orig"][i]
        mergefit_stats["ks_diff"][i] = mergefit_stats["ks"][i][0] - mergefit_stats["ks_orig"][i][0]
        
        #8) plot
        j = i // n_cols
        k = i % n_cols
        axs[j,k].plot(t_test,p_bayes, color = "steelblue", alpha = 1)
        axs[j,k].plot(t_test,p_bayes_orig, color = "orange", alpha = 1)
        axs[j,k].hist(t_op_orig, density = True, bins = 50, color = "green", alpha = 0.5)
        if mergefit_stats["ks_diff"][i] > 0.1:
            axs[j,k].set_title(edge    + "\nCRPS_merge: " + str(np.round(mergefit_stats["crps"][i],3)) 
                                   + "   |   KS_merge: " + str(np.round(mergefit_stats["ks"][i][0],3)) 
                                   + "\nCRPS_orig: " + str(np.round(mergefit_stats["crps_orig"][i],3)) 
                                   + "   |   KS_orig: " + str(np.round(mergefit_stats["ks_orig"][i][0],3)) 
                                   , fontsize = 15, color = "r")
        else:
            axs[j,k].set_title(edge    + "\nCRPS_merge: " + str(np.round(mergefit_stats["crps"][i],3)) 
                                       + "   |   KS_merge: " + str(np.round(mergefit_stats["ks"][i][0],3)) 
                                       + "\nCRPS_orig: " + str(np.round(mergefit_stats["crps_orig"][i],3)) 
                                       + "   |   KS_orig: " + str(np.round(mergefit_stats["ks_orig"][i][0],3)) 
                                       , fontsize = 15, color = "k")
        axs[j,k].legend(["mergefit","original fit", "observed"])
    toc = time.time()
    print("Time taken:", toc-tic, "secs")
    fig.savefig("mergefit_crps_ks.pdf")
    
    return mergefit_stats