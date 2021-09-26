# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 19:02:36 2021

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
from evaluate_length_ensemble import *
from select_data_edge import *

def evaluate_length_ensemble_obs(edge,df_test, df_train, 
                    filename_test = "aaf_map.yaml",filename_train = "tsc_map.yaml",
                    metric = "difference", cutoff = 100, max_obs = [10,100,1000],n_repeats = 5,
                    plot_graph = False, verbose = False,
                    random_state = None):
    """
    Wrapper for random_prior.py & evaluate_prior.py
    Max_obs is a vector of INTs
    
    Returns a dictionary: ks_dict
        Key is max_obs
        Values is corresponding ks_random

    """
    ks_dict = {}  
    
    for i in range(0,len(max_obs)):
        n = max_obs[i]
        ks_length, n_length, std_length = evaluate_length_ensemble(edge,
                    df_test = df_test, 
                    df_train = df_train, 
                    filename_test = filename_test,
                    filename_train = filename_train,
                    n_repeats = n_repeats,
                    cutoff = cutoff, 
                    metric = metric, max_obs = n,
                    plot_graph = plot_graph, verbose = verbose,
                    random_state = random_state)
        
        ks_dict[str(n)] = ks_length
    return ks_dict, n_length