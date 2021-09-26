# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 21:01:05 2021

@author: pyliu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 20:26:50 2021

@author: pyliu
"""

import pandas as pd
import numpy as np
import scipy as sp
import time
import random

from get_length import *
from select_data_edge import *
from evaluate_specific_prior import *

def start_ks_random(df_train, df_test, 
                    filename_train = "tsc_map.yaml", filename_test = "aaf_map.yaml",
                    metric = "difference",
                    cutoff = 1, verbose = False):
    """
    Get list of ks scores for most similar training edge for each test edge
    Similarity according to length

    Parameters
    ----------
    df_train : pandas df
        Observations in training edge
    df_test : pandas df
        Observations in test edge
    filename_train : STR
        Filename of training map. The default is "tsc_map.yaml".
    filename_train : STR
        Filename of test map. The default is "aaf_map.yaml".
    cutoff : INT
        Minimum observations in test and selected training edges. The default is 1.
    verbose : BOOL
        If true, print timing information. The default is False.

    Returns
    -------
    ks_list : FLOAT, vector
        KS value for each test edge (compared to most similar training edge)
    edge_test_list : STR, vector
        list of edges in test map wth more than cutoff datapoints
    edge_prior_list : STR, vector
        corresponding most similar edge from training map

    """
    if metric not in ["difference", "operation_time"]:
        raise ValueError("Invalid metric")
    tic = time.time()
    #1) Get context
    length_test = get_length(filename_test, suppress_message=True)
    length_train = get_length(filename_train, suppress_message=True)
    
    #2) Iniialise vars
    ks_list = []
    edge_list = list(length_test.keys())
    edges_train = list(length_train.keys())
    edge_test_list = []
    edge_prior_list = []
    
    #3) For each edge in training map with >= cutoff datapoints
    for i, edge_test in enumerate(edge_list):
        if i % 25 == 0 and verbose == True:
            toc = time.time()
            print(f"{i} iterations: {toc-tic} secs")
            
        subset_train = select_data_edge(df_test,edge_test)
        if len(subset_train) >= cutoff:
            #4) Select most similar edge
            index = random.randint(0,len(edges_train)-1)
            edge_train = edges_train[index]
            subset_train = select_data_edge(df_train, edge_train)
            while len(subset_train) < cutoff:
                edges_train.pop(index)
                if len(edges_train) < 1:
                    raise ValueError("Error: no data")
                index = random.randint(0,len(edges_train)-1)
                edge_train = edges_train[index]
                subset_train = select_data_edge(df_train, edge_train)
            edge_test_list.append(edge_test)
            edge_prior_list.append(edge_train)
            
            #5) Calculate ks
            ks = evaluate_specific_prior(edge_train = edge_train, edge_test = edge_test,
                                    df_train = df_train, df_test = df_test, 
                                    metric = metric, 
                                     prior_params = [1,0.1,1,1], max_obs = None, 
                                     random_state = None)
            ks_list.append(ks)
    
    if verbose == True:
        toc = time.time()
        print(toc-tic)
    
    return np.array(ks_list), edge_test_list, edge_prior_list