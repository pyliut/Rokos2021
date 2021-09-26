# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 20:26:50 2021

@author: pyliu
"""

import pandas as pd
import numpy as np
import scipy as sp
import time

from get_length import *
from select_data_edge import *
from evaluate_specific_prior import *
from similar_length_fast import *

def start_ks_length(df_train, df_test, 
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
    length_train = get_length(filename_train, suppress_message=True)
    length_test = get_length(filename_test, suppress_message=True)
    
    #2) Iniialise vars
    ks_list = []
    edge_list = list(length_test.keys())
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
            edge_train = similar_length_fast(edge_test, df_train = df_train,length_train=length_train, length_test=length_test, cutoff = cutoff)            
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