# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:27:02 2021

@author: pyliu
"""

import pandas as pd
import numpy as np
import scipy as sp
import time
import random

from random_prior import *
from specific_prior import *
from evaluate_prior import *
def evaluate_random_obs(edge,df_test, df_train, filename_train = "tsc_map.yaml",
                    metric = "difference", cutoff=100, max_obs = [10,100,250,500],
                    plot_graph = False, verbose = True,
                    random_state = None):
    """
    Wrapper for random_prior.py & evaluate_prior.py
    Max_obs is a vector of INTs
    
    Returns a dictionary: ks_dict
        Key is max_obs
        Values is corresponding ks_random

    """
    ks_dict = {}
    
    n = max_obs[0]
    mean_test, var_test, prior, t_op_prior, edge_prior = random_prior(df_train,
                                                filename = filename_train,metric = metric, 
                                                cutoff=cutoff, max_obs = n, 
                                                plot_graph = plot_graph,
                                                random_state = random_state)
    
    ks_random, n_random = evaluate_prior(edge, df_test, 
                       mean_test, var_test, prior, t_op_prior,
                       metric = metric, verbose = verbose,
                       random_state = random_state)
    
    ks_dict[str(n)] = ks_random
    
    
    for i in range(1,len(max_obs)):
        n = max_obs[i]
        mean_test, var_test, prior, t_op_prior, edge_prior = specific_prior(edge_prior,df_train,
                                                 metric = metric, max_obs = n, 
                                                 plot_graph = plot_graph,
                                                 random_state = random_state)
    
        ks_random, n_random = evaluate_prior(edge, df_test, 
                           mean_test, var_test, prior, t_op_prior,
                           metric = metric, verbose = verbose,
                           random_state = random_state)
        
        ks_dict[str(n)] = ks_random
    return ks_dict, n_random