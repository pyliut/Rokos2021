# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 19:52:31 2021

@author: pyliu
"""
import pandas as pd
import numpy as np
import math
import scipy as sp
from copy import copy
import time


from Lognormal import *
from Normal_Gamma_bayes import *
from integrate_pdf import *
from error_crps import *
from select_data_edge import *


def crps_between_edges(df, edge_list, metric = "median"):
    """
    Compares CRPS of a fitted distribution for one edge with observations from a second edge

    Parameters
    ----------
    df : pandas dataframe
        columns = ["origin", "target", "edge_id", "operation_time"]
    edge_list : STR, vector
        list of "edge_id" in a particular cluster
    metric : STR
        CRPS vector is as long as the number of observations of the second edge.
        We return either the ["mean", "variance", "min", "max"] of these CRPS scores.
        The default is "median".

    Raises
    ------
    ValueError
        The metric is invalid. metric can only be one of "mean" or "variance"

    Returns
    -------
    crps_df : pandas dataframe
        column & row headers are the edge_ids
        The dataframe contains the crps scores between the different edges

    """
    valid_metrics = ["mean", "median"]
    tic = time.time()
    if metric not in valid_metrics:
        raise ValueError("Invalid metric - check help(crps_between_edges) for valid metrics")
    # 1) Initialise pandas df
    #turn off warnings
    pd.options.mode.chained_assignment = None  # default='warn'
    crps_df = pd.DataFrame(index = edge_list, columns = edge_list)
    
    # 2) crps between edges
    for i in range(len(edge_list)-1):
        
        #2a) fit parameters of edge 1
        edge1 = edge_list[i]
        subset1 = select_data_edge(df, edge1)
        #independent variable to plot over
        t_op1 = subset1["operation_time"]
        
        offset1= np.min(t_op1) - 0.01
        t_log = np.log(t_op1 - offset1)
        #n_terms = 1 if you want just the initial estimates
        n_terms = len(t_log)    
        t_obs = t_log[0:n_terms]
        
        #set parameters
        mu_0 = 1
        beta = 0.1
        a = 1
        b = 1
        
        #Bayesian MAP estimate of mean & variance of Gaussian distribution
        mean_test1, var_test1, posterior1, mean_map1, var_map1 = Normal_Gamma_bayes(t_obs, mu_0, beta, a, b, plot_graph = False)
        
        # 2b) create models of edge 1
        precision = 2
        t_start= 10**(-precision)
        t_stop = ( (np.max(t_op1)) //5)*5 + 5    #round up to nearest 5 secs
        t_step = 10**(-precision)
        t_test = np.arange(t_start,t_stop,t_step)
        
        #Duration distribution using MAP parameters from Bayesian method
        p_bayes1 = Lognormal(t_test,mean_map1,var_map1)
        
        #account for offset
        t_offset1 = np.arange(t_stop, t_stop+offset1, t_step)
        p_offset1 = np.zeros(len(t_offset1))
        p_bayes1 = np.array([*p_offset1, *p_bayes1])
        t_test1 = np.array([*t_test, *t_offset1])
        
        
        for j in range(i+1,len(edge_list)):
            #2c) fit parameters of edge 2
            edge2 = edge_list[j]
            subset2 = select_data_edge(df, edge2)
            #independent variable to plot over
            t_op2 = subset2["operation_time"]
            
            offset2= np.min(t_op2) - 0.01
            t_log = np.log(t_op2 - offset2)
            
            #n_terms = 1 if you want just the initial estimates
            n_terms = len(t_log)    
            t_obs = t_log[0:n_terms]
            
            #set parameters
            mu_0 = 1
            beta = 0.1
            a = 1
            b = 1
            
            #Bayesian MAP estimate of mean & variance of Gaussian distribution
            mean_test2, var_test2, posterior2, mean_map2, var_map2 = Normal_Gamma_bayes(t_obs, mu_0, beta, a, b, plot_graph = False)

            # 2d) create models of edge 2
            #Duration distribution using MAP parameters from Bayesian method
            p_bayes2 = Lognormal(t_test,mean_map2,var_map2)
        
            #account for offset
            t_offset2 = np.arange(t_stop, t_stop+offset2, t_step)
            p_offset2 = np.zeros(len(t_offset2))
            p_bayes2 = np.array([*p_offset2, *p_bayes2])
            t_test2 = np.array([*t_test, *t_offset2])
    
            # 3) calculate crps & store result
            
            #fitted on edge 1 & tested against data from edge 1
            cdf_bayes1 = integrate_pdf(p_bayes1, spacing = t_test1[1] - t_test1[0])
            crps1 = error_crps(np.array(t_op1),cdf_bayes1,t_test1, method = "rectangle")
            if metric == "mean":
                crps_df.loc[edge1,edge1] = crps1["crps"].mean()
            elif metric == "median":
                crps_df.loc[edge1,edge1] = crps1["crps"].median()
            elif metric == "min":
                crps_df.loc[edge1,edge1] = crps1["crps"].min()
            elif metric == "max":
                crps_df.loc[edge1,edge1] = crps1["crps"].max()
            else:
                raise ValueError("Invalid metric - check help(crps_between_edges) for valid metrics")
    
            #fitted on edge 2 & tested against data from edge 1
            cdf_bayes2 = integrate_pdf(p_bayes2, spacing = t_test2[1] - t_test2[0])
            crps2 = error_crps(np.array(t_op1),cdf_bayes2,t_test2, method = "rectangle")
            if metric == "mean":
                crps_df.loc[edge2,edge1] = crps2["crps"].mean()
            elif metric == "median":
                crps_df.loc[edge2,edge1] = crps2["crps"].median()
            elif metric == "min":
                crps_df.loc[edge2,edge1] = crps1["crps"].min()
            elif metric == "max":
                crps_df.loc[edge2,edge1] = crps1["crps"].max()
            else:
                raise ValueError("Invalid metric - check help(crps_between_edges) for valid metrics")
    
            #fitted on edge 1 & tested against data from edge 2
            cdf_bayes1 = integrate_pdf(p_bayes1, spacing = t_test1[1] - t_test1[0])
            crps1 = error_crps(np.array(t_op2),cdf_bayes1,t_test1, method = "rectangle")
            if metric == "mean":
                crps_df.loc[edge1,edge2] = crps1["crps"].mean()
            elif metric == "median":
                crps_df.loc[edge1,edge2] = crps1["crps"].median()
            elif metric == "min":
                crps_df.loc[edge1,edge2] = crps1["crps"].min()
            elif metric == "max":
                crps_df.loc[edge1,edge2] = crps1["crps"].max()
            else:
                raise ValueError("Invalid metric - check help(crps_between_edges) for valid metrics")
                
            #fitted on edge 2 & tested against data from edge 2
            cdf_bayes2 = integrate_pdf(p_bayes2, spacing = t_test2[1] - t_test2[0])
            crps2 = error_crps(np.array(t_op2),cdf_bayes2,t_test2, method = "rectangle")
            if metric == "mean":
                crps_df.loc[edge2,edge2] = crps2["crps"].mean()
            elif metric == "median":
                crps_df.loc[edge2,edge2] = crps2["crps"].median()
            elif metric == "min":
                crps_df.loc[edge2,edge2] = crps1["crps"].min()
            elif metric == "max":
                crps_df.loc[edge2,edge2] = crps1["crps"].max()
            else:
                raise ValueError("Invalid metric - check help(crps_between_edges) for valid metrics")
                
    toc = time.time()
    print("Time taken:", toc-tic, "secs")
    return crps_df
            