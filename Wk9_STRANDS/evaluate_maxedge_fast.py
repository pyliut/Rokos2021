# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 22:10:20 2021

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

def evaluate_maxedge_fast(df,clusters,fit, metric = "difference", n_fitted = 2):
    """
    Returns mean KS score within cluster compared to edge with most datapoints in cluster
    Faster implementation than evaluate_maxedge since we can look up the fitting params from fit 

    Parameters
    ----------
    df : pandas df
        observations. columns are ["origin", "target", "edge_id", "time_to_waypoint", "operation_time"]
    clusters : pandas df
        columns are ["edge_id", "cluster_id"]
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
            s1 = float(fit[fit["edge_id"] == edge1]["s"])
            loc1 = float(fit[fit["edge_id"] == edge1]["loc"])
            scale1 = float(fit[fit["edge_id"] == edge1]["scale"])
            
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