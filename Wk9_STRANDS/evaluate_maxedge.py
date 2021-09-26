# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 21:59:43 2021

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

def evaluate_maxedge(df,clusters, metric = "difference", n_fitted = 2):
    """
    Returns mean KS score within cluster compared to edge with most datapoints in cluster

    Parameters
    ----------
    df : pandas df
        observations. columns are ["origin", "target", "edge_id", "time_to_waypoint", "operation_time"]
    clusters : pandas df
        columns are ["edge_id", "cluster_id"]
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
        invalid metric or invalid n_fitted

    Returns
    -------
    ks_df : pandas df
        columns = ["cluster_id", "n_edges", "ks_mean", "ks_median", "ks_max", "ks_min"]

    """
    
    tic = time.time()
    clusters["n_obs"] = -1
    
    #augment with count data & order by no. of observations
    for i in range(len(clusters)):
        subset = select_data_edge(df,clusters["edge_id"][i])
        clusters["n_obs"][i] = int(len(subset))
    
    clusters = clusters.sort_values("n_obs",ascending = False).reset_index(drop = True)
    
    #turn off warnings
    pd.options.mode.chained_assignment = None  # default='warn'
    
    #initialise output
    ks_df = pd.DataFrame(index = np.arange(clusters["cluster_id"].max() + 1), 
                         columns = ["cluster_id","n_edges","ks_mean","ks_median","ks_max","ks_min"])
    
    ks_all = []
    
    for i in range(clusters["cluster_id"].max() + 1):
        #initialise list to store ks
        ks_list = []
        
        #isolate cluster
        current_cluster = clusters[clusters["cluster_id"]==i]
        current_cluster = current_cluster.reset_index(drop = True) 
        toc = time.time()
        print("cluster",i,"(",len(current_cluster), "edges ):", toc-tic, "secs")
        
        #fit edge with most data
        # Select data of interest
        edge1 = current_cluster["edge_id"][0]
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
            #use offset and take log of data
            offset1= np.min(t_op1) - 0.01
            t_obs1 = np.log(t_op1 - offset1)
    
            #set parameters
            mu_0 = 1
            beta = 0.1
            a = 1
            b = 1
    
            #Bayesian MAP estimate of mean & variance of Gaussian distribution
            mean_test1, var_test1, posterior1, mean_map1, var_map1 = Normal_Gamma_bayes(t_obs1, mu_0, beta, a, b, plot_graph = False)
    
        #fit other edges
        for j in range(1,len(current_cluster)):
            #fit other edges in cluster
            # Select data of interest
            edge2 = current_cluster["edge_id"][j]
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
                ks,p_val = sp.stats.kstest(t_op2, lambda k: sp.stats.lognorm.cdf( k, s = np.sqrt(var_map1), loc = offset1, scale = np.exp(mean_map1)) )
            elif n_fitted == 2:
                #fit second edge
                #use offset and take log of data
                offset2= np.min(t_op2) - 0.01
                t_obs2 = np.log(t_op2 - offset2)
    
                #set parameters
                mu_0 = 1
                beta = 0.1
                a = 1
                b = 1
    
                #Bayesian MAP estimate of mean & variance of Gaussian distribution
                mean_test2, var_test2, posterior2, mean_map2, var_map2 = Normal_Gamma_bayes(t_obs2, mu_0, beta, a, b, plot_graph = False)
    
                #calculate ks
                precision = 2
                t_start= 10**(-precision)
                t_stop = np.max( [np.max(t_op1), np.max(t_op2)] )  
                t_step = 10**(-precision)
                t_test = np.arange(t_start,t_stop,t_step)
            
                cdf1 = sp.stats.lognorm.cdf( t_test, s = np.sqrt(var_map1), loc = offset1, scale = np.exp(mean_map1) ) 
                cdf2 = sp.stats.lognorm.cdf( t_test, s = np.sqrt(var_map2), loc = offset2, scale = np.exp(mean_map2) ) 
                ks = np.abs( np.max(cdf1 - cdf2) )
            else:
                raise ValueError("Invalid n_fitted. Validn_fitted is [0,1,2]")
            ks_list.append(ks)
            ks_all.append(ks)
            
        #store stats
        ks_df["cluster_id"][i] = i
        ks_df["n_edges"][i] = len(current_cluster)
        if len(ks_list) > 0:
            ks_df["ks_mean"][i] = np.mean(ks_list)
            ks_df["ks_median"][i] = np.median(ks_list)
            ks_df["ks_max"][i] = np.max(ks_list)
            ks_df["ks_min"][i] = np.min(ks_list)
        else:
            ks_df["ks_mean"][i] = np.NaN
            ks_df["ks_median"][i] = np.NaN
            ks_df["ks_max"][i] = np.NaN
            ks_df["ks_min"][i] = np.NaN
    
    toc = time.time()
    print("n_fitted:", n_fitted)
    print("Time taken:", toc-tic, "secs")
    print("Mean KS:",np.mean(ks_all))
        
    return ks_df