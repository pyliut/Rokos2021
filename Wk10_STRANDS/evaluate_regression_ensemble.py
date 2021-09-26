# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 17:42:45 2021

@author: pyliu
"""


import scipy as sp
import numpy as np
import pandas as pd
import random

import time

from similar_regression import *
from specific_prior import *
from evaluate_prior import *
from select_data_edge import *
from update_mean import *
from update_var import *

def evaluate_regression_ensemble(edge,clf,df_test, df_train, 
                    filename_test = "aaf_map.yaml",filename_train = "tsc_map.yaml",
                    metric = "difference", cutoff = 100, max_obs = 10,n_repeats = 5,
                    n_similar = 10, random_state = None,
                    plot_graph = False, verbose = True):
    """
    Wrapper for similar_length.py, specific_prior.py & evaluate_prior.py

    Additional Parameters
    ----------
    n_repeats : INT
        number of iterations in ensemble. The default is 5.

    """
    
    tic = time.time()
    similar_edges, df_predict = similar_regression(edge, clf = clf, 
                           filename_test = filename_test, filename_train = filename_train, 
                           n_similar = n_similar)
    
    index = 0
    for i in range(len(similar_edges)):
        edge_prior = similar_edges[i]
        subset = select_data_edge(df_train,edge_prior)
        if len(subset) >= cutoff:
            index = i
            break
        edge_prior = similar_edges[0]
    print("Similar edge:", edge_prior)
    print("length diff:", df_predict["edge_length_diff"][index])
    print("n_obs:", len(subset))
    
    #create seeds
    random.seed(random_state)
    seeds = [random.randint(0,1024) for i in range(n_repeats)]
    
    #initial run
    mean_test, var_test, prior, t_op_prior, edge_prior = specific_prior(edge_prior,df_train,
                                                 metric = metric, max_obs = max_obs, 
                                                 plot_graph = plot_graph, random_state = seeds[0])
    
    ks_list, n_list = evaluate_prior(edge, df_test, 
                       mean_test, var_test, prior, t_op_prior,
                       metric = metric, verbose = False,
                       random_state = seeds[0])
    ks_list = np.array(ks_list)
    n_list = np.array(n_list)
    std_list = update_var(ks_list,0,0,ks_list)
    
    #additional runs
    for i in range(1,n_repeats):
        if (i+1) % 2 == 0 and verbose == True:
            toc = time.time()
            print(i+1,"iterations:",toc-tic,"secs")
            
        mean_test, var_test, prior, t_op_prior, edge_prior = specific_prior(edge_prior,df_train,
                                                 metric = metric, max_obs = max_obs, 
                                                 plot_graph = plot_graph,
                                                 random_state = seeds[i])
    
        ks_current, n_current = evaluate_prior(edge, df_test, 
                       mean_test, var_test, prior, t_op_prior,
                       metric = metric, verbose = False,
                       random_state = seeds[i])
        ks_list = update_mean(ks_list,i,ks_current)
        std_list = update_var(ks_list,std_list,i,ks_current)
        
    #find std
    std_list = np.sqrt(std_list)
    
    toc = time.time()
    print("Time taken (evaluate_regression_ensemble):",toc-tic,"secs")
    return ks_list,n_list,std_list