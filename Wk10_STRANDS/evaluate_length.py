# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 18:07:26 2021

@author: pyliu
"""


import pandas as pd
import numpy as np
import scipy as sp
import time
import random

from similar_length import *
from specific_prior import *
from evaluate_prior import *
from select_data_edge import *
def evaluate_length(edge,df_test, df_train, 
                    filename_test = "aaf_map.yaml",filename_train = "tsc_map.yaml",
                    metric = "difference", cutoff = 100, max_obs = 10,
                    plot_graph = False, verbose = False,
                    random_state = None):
    """
    Wrapper for similar_length.py, specific_prior.py & evaluate_prior.py
    Max_obs is a scalar INT

    """
    
    similar_edges, length_diff = similar_length(edge, filename_test = "aaf_map.yaml", filename_train = "tsc_map.yaml", n_similar=5)
    index = 0
    for i in range(len(similar_edges)):
        edge_prior = similar_edges[i]
        subset = select_data_edge(df_train,edge_prior)
        if len(subset) >= cutoff:
            index = i
            break
        edge_prior = similar_edges[0]
    print("Similar edge:", edge_prior)
    print("length diff:", length_diff[index])
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
