# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 22:39:32 2021

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
from Normal_Gamma_bayes_initial import *

def evaluate_bayes(edge, df, metric = "difference", 
                   prior_params = [1,0.1,1,1], verbose = False,
                   random_state = None):
    """
    Progressively add more data to Bayesian fit (naive prior)

    Parameters
    ----------
    edge : STR
        edge name
    df : pandas df
        observations
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
    
    
    #2) Get data
    subset = select_data_edge(df,edge)
    subset = subset.sample(frac=1, random_state = random_state).reset_index(drop=True)
    if metric == "operation_time":
        t_op = subset["operation_time"]
    elif metric == "difference":
        t_op = subset["operation_time"] - subset["time_to_waypoint"]
    else:
        raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")

    
    #3) initial params
    offset = t_op[0]-0.01
    t_log = np.log(t_op - offset)
    #Bayesian MAP estimate of mean & variance of Gaussian distribution
    mean_test, var_test, posterior, mean_map, var_map = Normal_Gamma_bayes_initial(t_log, 
                                                        mu_0, beta, a, b, 
                                                        n_initial = 1, plot_graph = False)     
    
    #standard form
    s = np.sqrt(var_map)
    loc = offset
    scale = np.exp(mean_map)
    
    #ks
    ks,pval = sp.stats.kstest(t_op, 
                    lambda k: sp.stats.lognorm.cdf( k, s = s, loc = loc, scale = scale) )
    ks_list.append(ks)
    n_list.append(1)
    

    if len(subset) > 1:
        for i in range(1,len(subset)):
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
                t_log = np.log(t_op - offset)
                t_obs = t_log[:i+1]
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










def evaluate_bayes_legacy1(edge,df,metric = "difference", max_n = 100,
                        prior_params = [1,0.1,1,1], verbose = False):
    """
    Progressively add more data to Bayesian fit

    Parameters
    ----------
    edge : STR
        edge name
    df : pandas df
        observations
    metric : STR
        Valid metrics are ["operation_time","difference"]. The default is "difference".
    prior_params : FLOAT, vector
        4-element array. Elements are [mu_0, beta, a , b]. The default is [1,0.1,1,1].
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
    
    #set parameters for Bayesian fit
    mu_0 = prior_params[0]
    beta = prior_params[1]
    a = prior_params[2]
    b = prior_params[3]
    
    #2) Get data
    subset = select_data_edge(df,edge)
    subset = subset.sample(frac=1).reset_index(drop=True)
    if metric == "operation_time":
        t_op = subset["operation_time"]
    elif metric == "difference":
        t_op = subset["operation_time"] - subset["time_to_waypoint"]
    else:
        raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")

    
    #3) initial params
    offset = np.min(t_op)-0.01
    t_log = np.log(t_op - offset)
    #Bayesian MAP estimate of mean & variance of Gaussian distribution
    mean_test, var_test, posterior, mean_map, var_map = Normal_Gamma_bayes_initial(t_log, mu_0, beta, a, b, plot_graph = False)     
    
    #standard form
    s = np.sqrt(var_map)
    loc = offset
    scale = np.exp(mean_map)
    
    #ks
    ks,pval = sp.stats.kstest(t_op, 
                lambda k: sp.stats.lognorm.cdf( k, s = s, loc = loc, scale = scale) )
    ks_list.append(ks)
    n_list.append(1)
    
    
    for i in range( 2,np.min([len(subset),max_n]) ):
        if i % 200==0 and verbose == True:
            toc = time.time()
            print(i, "iterations:", toc-tic, "secs")
    
        #3) fit
        t_obs = t_log[:i+1]
        #Bayesian MAP estimate of mean & variance of Gaussian distribution
        mean_test, var_test, posterior, mean_map, var_map = Normal_Gamma_bayes_update(mean_test, var_test, 
                                                                                      posterior, t_obs, threshold = 0.9, plot_graph = False)
        s = np.sqrt(var_map)
        loc = offset
        scale = np.exp(mean_map)
        
        #4) calculate KS stat
        ks,pval = sp.stats.kstest(t_op, 
                lambda k: sp.stats.lognorm.cdf( k, s = s, loc = loc, scale = scale) )
        
        ks_list.append(ks)
        n_list.append(i+1)
    
    return ks_list, n_list  





def evaluate_bayes_legacy2(edge,df,metric = "difference", max_n = 100,
                        prior_params = [1,0.1,1,1], verbose = False):
    """
    Progressively add more data to Bayesian fit

    Parameters
    ----------
    edge : STR
        edge name
    df : pandas df
        observations
    metric : STR
        Valid metrics are ["operation_time","difference"]. The default is "difference".
    prior_params : FLOAT, vector
        4-element array. Elements are [mu_0, beta, a , b]. The default is [1,0.1,1,1].
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
    
    #set parameters for Bayesian fit
    mu_0 = prior_params[0]
    beta = prior_params[1]
    a = prior_params[2]
    b = prior_params[3]
    
    #2) Get data
    subset = select_data_edge(df,edge)
    subset = subset.sample(frac=1).reset_index(drop=True)
    if metric == "operation_time":
        t_op = subset["operation_time"]
    elif metric == "difference":
        t_op = subset["operation_time"] - subset["time_to_waypoint"]
    else:
        raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")

    for i in range( np.min([len(subset),max_n]) ):
        if i % 20==0 and verbose == True:
            toc = time.time()
            print(i, "iterations:", toc-tic, "secs")
    
        #3) fit
        t_obs = t_op[:i+1]
        offset= np.min(t_obs) - 0.01
        t_obs = np.log(t_obs - offset)
        #Bayesian MAP estimate of mean & variance of Gaussian distribution
        mean_test, var_test, posterior, mean_map, var_map = Normal_Gamma_bayes(t_obs, mu_0, beta, a, b, plot_graph = False)
        s = np.sqrt(var_map)
        loc = offset
        scale = np.exp(mean_map)
        
        #4) calculate KS stat
        ks,pval = sp.stats.kstest(t_op, 
                lambda k: sp.stats.lognorm.cdf( k, s = s, loc = loc, scale = scale) )
        
        ks_list.append(ks)
        n_list.append(i+1)
    
    return ks_list, n_list  