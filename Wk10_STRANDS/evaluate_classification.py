# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:40:41 2021

@author: pyliu
"""


import pandas as pd
import numpy as np
import scipy as sp
import time
import random

from similar_classification import *
from specific_prior import *
from evaluate_prior import *
from select_data_edge import *
def evaluate_classification(edge,clf, df_test, df_train, 
                    filename_test = "aaf_map.yaml",filename_train = "tsc_map.yaml",
                    metric = "difference", plot_graph = False, verbose = False, 
                    cutoff = 100, max_obs = 10,n_similar = 5,
                    random_state = None):
    """
    Wrapper for similar_classification.py, specific_prior.py & evaluate_prior.py
    Max_obs is a scalar INT

    """
    
    similar_edges, df_predict = similar_classification(edge, clf = clf, 
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

    mean_test, var_test, prior, t_op_prior, edge_prior = specific_prior(edge_prior,df_train,
                                                 metric = metric, max_obs = max_obs, 
                                                 plot_graph = plot_graph,
                                                 random_state = random_state)
    
    ks_length, n_length = evaluate_prior(edge, df_test, 
                       mean_test, var_test, prior, t_op_prior,
                       metric = metric, verbose = verbose,
                       random_state = random_state)
    
    return ks_length, n_length
