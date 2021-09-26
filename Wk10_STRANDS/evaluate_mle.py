# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:29:54 2021

@author: pyliu
"""
import pandas as pd
import numpy as np
import scipy as sp
import time

from select_data_edge import *
from Gaussian_ml import *
from update_mean import *
from update_var import *

def evaluate_mle(edge, df, metric = "difference", verbose = False,
                 random_state = None):
    """
    Progressively add more data to MLE fit

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
    
    #2) Get data
    subset = select_data_edge(df,edge)
    subset = subset.sample(frac=1,random_state=random_state).reset_index(drop=True)
    if metric == "operation_time":
        t_op = subset["operation_time"]
    elif metric == "difference":
        t_op = subset["operation_time"] - subset["time_to_waypoint"]
    else:
        raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")

    
    #3) initial params
    offset = t_op[0]-0.01
    t_log = np.log(t_op - offset)
    mean = 0.01
    var = 0
    
    #standard form
    s = np.sqrt(var)
    loc = offset
    scale = np.exp(mean)
    
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
                mean = update_mean(mean,i,t_log[i])
                var = update_var(mean,var,i,t_log[i])
                #standard form
                s = np.sqrt(var)
                scale = np.exp(mean)
                
            else:
                offset = t_op[i]-0.01
                t_obs = t_op[:i+1]
                s,loc,scale = Gaussian_ml(t_obs)
                t_log = np.log(t_op - loc)
                n_resets += 1
                
                #return to non-standard form
                var = s*s
                mean = np.log(scale)
                

            #calculate KS stat
            ks,pval = sp.stats.kstest(t_op, 
                    lambda k: sp.stats.lognorm.cdf( k, s = s, loc = loc, scale = scale) )

            ks_list.append(ks)
            n_list.append(i+1)
    
    if verbose == True:
        print("No. of resets for offset:",n_resets)
    
    return ks_list, n_list











def evaluate_mle_legacy(edge, df, metric = "difference", verbose = False):
    """
    Progressively add more data to MLE fit

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
    
    #2) Get data
    subset = select_data_edge(df,edge)
    subset = subset.sample(frac=1).reset_index(drop=True)
    if metric == "operation_time":
        t_op = subset["operation_time"]
    elif metric == "difference":
        t_op = subset["operation_time"] - subset["time_to_waypoint"]
    else:
        raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")

    for i in range(len(subset)):
        if i % 200==0 and verbose == True:
            toc = time.time()
            print(i, "iterations:", toc-tic, "secs")
    
        #3) fit
        t_obs = t_op[:i+1]
        s,loc,scale = Gaussian_ml(t_obs)
        
        #4) calculate KS stat
        ks,pval = sp.stats.kstest(t_op, 
                lambda k: sp.stats.lognorm.cdf( k, s = s, loc = loc, scale = scale) )
        
        ks_list.append(ks)
        n_list.append(i+1)
    
    return ks_list, n_list
            
        
        
    
    