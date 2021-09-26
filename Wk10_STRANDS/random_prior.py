# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 12:03:13 2021

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



def random_prior(df,filename = "aaf_map.yaml", metric = "difference", cutoff=20, 
                 prior_params = [1,0.1,1,1], max_obs = 10, 
                 plot_graph = False, verbose = False,
                 random_state = None):

    edges = df["edge_id"].unique()
    n_selections = 0
    while n_selections < 20:
        select_index = random.randint(0,len(edges)-1)
        edge = edges[select_index]
        subset = select_data_edge(df, edge)
        n_selections += 1

        if len(subset) >= cutoff:
            break
        
    if verbose == True:
        print("Selected edge:", edge)
        print("no. of datapoints:", len(subset))

    if len(subset) < max_obs:
        n_sample = len(subset)
    else:
        n_sample = max_obs
        
    subset = subset.sample(n=n_sample,random_state=random_state).reset_index(drop=True)
    if metric == "operation_time":
        t_op = subset["operation_time"]
    elif metric == "difference":
        t_op = subset["operation_time"] - subset["time_to_waypoint"]
    else:
        raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")


    mu_0 = prior_params[0]
    beta = prior_params[1]
    a = prior_params[2]
    b = prior_params[3]

    offset = np.min(t_op)-0.01
    t_obs = np.log(t_op - offset)
    #Bayesian MAP estimate of mean & variance of Gaussian distribution
    mean_test, var_test, posterior, mean_map, var_map = Normal_Gamma_bayes_nothreshold(t_obs, 
                                                                                       mu_0, beta, a, b, 
                                                                                       max_range = 10,plot_graph = plot_graph)     

    return mean_test, var_test, posterior, t_op, edge