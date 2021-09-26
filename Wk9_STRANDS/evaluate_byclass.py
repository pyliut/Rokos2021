# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 16:55:12 2021

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

def evaluate_byclass(df,binclass,fit, metric = "difference", n_fitted = 2):
    """
    Returns mean KS score within cluster comparing all edges with all other edges
    Fast implementation since we can look up the fitting params from fit 

    Parameters
    ----------
    df : pandas df
        observations. columns are ["origin", "target", "edge_id", "time_to_waypoint", "operation_time"]
    binclass : pandas df
        columns are ["edge1","edge2", "cluster_id"]
    fit: pandas df
        lognorm params. columns are ["edge_id", "origin", "target", "n_obs",
                                     "s", "loc", "scale"]
    metric : STR
        Valid metrics are "difference" & "operation_time". The default is "difference".
    n_fitted : INT
        n_fitted = 0: use 2-sample KS test on raw observation data
        n_fitted = 1: edge with most data is fitted. Comparison edges are not
        n_fitted = 2: both edges are fitted using Bayesian lognormal optimisation
        The default is 0.

    Raises
    ------
    ValueError
        invalid metric, n_fitted or class

    Returns
    -------
    ks_df : pandas df
        columns = ["cluster_id", "n_edges", "ks_mean", "ks_median", "ks_max", "ks_min"]

    """
    
    tic = time.time()
    ks_list0 = []
    ks_list1 = []
    
    #augment with count data & order by no. of observations
    for i in range(len(binclass)):
        if i % 5000 == 0:
            toc = time.time()
            print(i,"iterations:", toc-tic, "secs")
    
        # Select data of interest
        edge1 = binclass["edge1"][i]
        subset1 = select_data_edge(df, edge1)
        #independent variable to plot over
        if metric == "difference":
            t_op1 = subset1["operation_time"] - subset1["time_to_waypoint"]
        elif metric == "operation_time":
            t_op1 = subset1["operation_time"]
        else:
            raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")
        
        if n_fitted == 1 or n_fitted == 2:
            #fit edge 1
            s1 = float(fit[fit["edge_id"] == edge1]["s"])
            loc1 = float(fit[fit["edge_id"] == edge1]["loc"])
            scale1 = float(fit[fit["edge_id"] == edge1]["scale"])
            
        #fit other edges
        # Select data of interest
        edge2 = binclass["edge2"][i]
        subset2 = select_data_edge(df, edge2)
        #independent variable to plot over
        if metric == "difference":
            t_op2 = subset2["operation_time"] - subset2["time_to_waypoint"]
        elif metric == "operation_time":
            t_op2 = subset2["operation_time"]
        else:
            raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")
        
        if n_fitted == 0:
            ks,p_val = sp.stats.ks_2samp(t_op1,t_op2)
        elif n_fitted == 1:
            ks,p_val = sp.stats.kstest(t_op2, lambda k: sp.stats.lognorm.cdf( k, s = s1, loc = loc1, scale = scale1) )
        elif n_fitted == 2:
            #fit edge 1
            s2 = float(fit[fit["edge_id"] == edge2]["s"])
            loc2 = float(fit[fit["edge_id"] == edge2]["loc"])
            scale2 = float(fit[fit["edge_id"] == edge2]["scale"])
            
            #calculate ks
            precision = 2
            t_start= 10**(-precision)
            t_stop = np.max( [np.max(t_op1), np.max(t_op2)] )  
            t_step = 10**(-precision)
            t_test = np.arange(t_start,t_stop,t_step)
        
            cdf1 = sp.stats.lognorm.cdf( t_test, s = s1, loc = loc1, scale = scale1 ) 
            cdf2 = sp.stats.lognorm.cdf( t_test, s = s2, loc = loc2, scale = scale2 ) 
            ks = np.abs( np.max(cdf1 - cdf2) )
        else:
            raise ValueError("Invalid n_fitted. Validn_fitted is [0,1,2]")
        
        if binclass["same_cluster"][i] == 0:
            ks_list0.append(ks)
        elif binclass["same_cluster"][i] == 1:
            ks_list1.append(ks)
        else:
            raise ValueError("Invalid Class: must be 0 or 1")
    
    toc = time.time()
    print("n_fitted:", n_fitted)
    print("Time taken:", toc-tic, "secs")
    print("n0:", len(ks_list0), "n1:", len(ks_list1))
    print("Mean KS (class 0):",np.mean(ks_list0))
    print("Mean KS (class 1):",np.mean(ks_list1))
        
    return np.mean(ks_list0),np.mean(ks_list1)