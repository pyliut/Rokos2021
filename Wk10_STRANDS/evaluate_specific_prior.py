# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 17:02:24 2021

@author: pyliu
"""

import scipy as sp
import numpy as np
import pandas as pd
import random

from select_data_edge import *
from Normal_Gamma_bayes_nothreshold import *
from Normal_Gamma_bayes_initial import *
from Normal_Gamma_bayes_updateone import *



def evaluate_specific_prior(edge_train, edge_test,
                            df_train, df_test, metric = "difference", 
                 prior_params = [1,0.1,1,1], max_obs = 10, 
                 random_state = None):
    
    #1) get data for training edge
    subset_train = select_data_edge(df_train, edge_train)
    #only use max_obs observations for training
    if max_obs == None:
        n_sample = len(subset_train)
    else:
        if len(subset_train) < max_obs:
            n_sample = len(subset_train)
        else:
            n_sample = max_obs
    
    subset_train = subset_train.sample(n=n_sample, random_state=random_state).reset_index(drop=True)
    if metric == "operation_time":
        t_op_train = subset_train["operation_time"]
    elif metric == "difference":
        t_op_train = subset_train["operation_time"] - subset_train["time_to_waypoint"]
    else:
        raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")

    #2) Train on training edge data
    mu_0 = prior_params[0]
    beta = prior_params[1]
    a = prior_params[2]
    b = prior_params[3]

    offset = np.min(t_op_train)-0.01
    t_obs = np.log(t_op_train - offset)
    #Bayesian MAP estimate of mean & variance of Gaussian distribution
    mean_test, var_test, posterior, mean_map, var_map = Normal_Gamma_bayes_nothreshold(t_obs, 
                                                                                       mu_0, beta, a, b, 
                                                                                       max_range = 10,plot_graph = False)     
    #standard form
    s = np.sqrt(var_map)
    loc = offset
    scale = np.exp(mean_map)
    
    #3) Get data for actual edge
    subset_test = select_data_edge(df_test,edge_test)
    if metric == "operation_time":
        t_op_test = subset_test["operation_time"]
    elif metric == "difference":
        t_op_test = subset_test["operation_time"] - subset_test["time_to_waypoint"]
    else:
        raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")
    
    #ks
    ks,pval = sp.stats.kstest(t_op_test, 
                    lambda k: sp.stats.lognorm.cdf( k, s = s, loc = loc, scale = scale) )

    return ks