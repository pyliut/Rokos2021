# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 12:59:04 2021

@author: pyliu
"""
import pandas as pd
import numpy as np
import scipy as sp
import time

from select_data_edge import *
from Gamma import *
from Lognormal import *
from Gaussian import *
from Gaussian_broadcast import *
from Normal_Gamma import *

from Normal_Gamma_bayes import *
from Normal_Gamma_bayes_update import *
from Normal_Gamma_bayes_updateone import *
from Normal_Gamma_bayes_nothreshold import *
from random_prior import *


def evaluate_prior(edge, df, 
                   mean_test, var_test, posterior, t_op_prior,
                   prior_params = [1,0.1,1,1],
                   metric = "difference", verbose = False,
                   random_state = None):
    """
    Progressively add more data to Bayesian fit (for a specified prior - i.e. the input variable called posterior)

    Parameters
    ----------
    edge : STR
        edge name
    df : pandas df
        observations
    filename : STR
        Name of file containing topological map data. The default is "aaf_map.yaml".
    metric : STR
        Valid metrics are ["operation_time","difference"]. The default is "difference".
    verbose : BOOL
        print progress information if True. The default is True.

    Raises
    ------
    ValueError
        Invalid metric

    Returns
    -------
    ks_list : FLOAT, vector
        ks statistic
    n_list : INT, vector
        number of datapoints used in fitting

    """
    
    tic = time.time()
    
    #1) initialise vars
    ks_list = []
    n_list = []
    n_resets = 0
    
    #set parameters for Bayesian fit
    mu_0 = prior_params[0]
    beta = prior_params[1]
    a = prior_params[2]
    b = prior_params[3]
    
    
    #2) Get data for actual edge
    subset = select_data_edge(df,edge)
    subset = subset.sample(frac=1,random_state=random_state).reset_index(drop=True)
    if metric == "operation_time":
        t_op = subset["operation_time"]
    elif metric == "difference":
        t_op = subset["operation_time"] - subset["time_to_waypoint"]
    else:
        raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")
        
    offset = np.min(t_op_prior)-0.01
    t_log = np.log(t_op - offset)
    
    
    #3) Find MAP value & KS value for prior
    p_max = np.amax(posterior)
    ind = np.argwhere(posterior == p_max)
    # ind[0,0] corresponds to index of MAP estimate of var
    # ind[0,1] corresponds to index of MAP estimate of mean  
    mean_map = mean_test[ind[0,1]]
    var_map = var_test[ind[0,0]]  
    
    #standard form
    s = np.sqrt(var_map)
    loc = offset
    scale = np.exp(mean_map)
    
    #ks
    ks,pval = sp.stats.kstest(t_op, 
                    lambda k: sp.stats.lognorm.cdf( k, s = s, loc = loc, scale = scale) )
    ks_list.append(ks)
    n_list.append(0)
    
    
    

    if len(t_op) > 0:
        for i in range(0,len(t_op)):
            if i % 200==0 and verbose == True:
                toc = time.time()
                print(i, "iterations:", toc-tic, "secs")
                
            
            #4) update params
            if t_op[i] - offset >= 0.01:
                t_obs = t_log[i]
                #Bayesian MAP estimate of mean & variance of Gaussian distribution
                mean_test, var_test, posterior, mean_map, var_map = Normal_Gamma_bayes_updateone(mean_test, var_test, 
                                                                                              posterior, t_obs, threshold = 0.9, plot_graph = False)
                #standard form
                s = np.sqrt(var_map)
                scale = np.exp(mean_map)
                
            else:
                offset = t_op[i]-0.01
                t_log_prior = np.array(np.log(t_op_prior - offset))
                t_log = np.array(np.log(t_op - offset))
                t_obs = np.array([*t_log_prior,*t_log[:i+1]])
                mean_test, var_test, posterior, mean_map, var_map = Normal_Gamma_bayes_initial(t_obs, 
                                                                    mu_0, beta, a, b, 
                                                                    n_initial = i, plot_graph = False)  
                n_resets += 1
                
                #standard form
                s = np.sqrt(var_map)
                scale = np.exp(mean_map)
                loc = offset
                

            #calculate KS stat
            ks,pval = sp.stats.kstest(t_op, 
                    lambda k: sp.stats.lognorm.cdf( k, s = s, loc = loc, scale = scale) )

            ks_list.append(ks)
            n_list.append(i+1)
    
    if verbose == True:
        print("No. of resets for offset:",n_resets)
    
    return ks_list, n_list