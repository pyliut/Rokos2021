# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:15:57 2021

@author: pyliu
"""
import pandas as pd
import numpy as np
import scipy as sp
import time
import random

from random_prior import *
from evaluate_prior import *
def evaluate_random(edge,df_test, df_train, filename_train = "tsc_map.yaml",
                    metric = "difference", cutoff=100, max_obs = 10,
                    plot_graph = False, verbose = True,
                    random_state = None):
    """
    Wrapper for random_prior.py & evaluate_prior.py
    Max_obs is a scalar INT

    """
    

    mean_test, var_test, prior, t_op_prior, edge_prior = random_prior(df_train,
                                            filename = filename_train,metric = metric, 
                                            cutoff=cutoff, max_obs = max_obs, 
                                            plot_graph = plot_graph,
                                            random_state = random_state)

    ks_random, n_random = evaluate_prior(edge, df_test, 
                       mean_test, var_test, prior, t_op_prior,
                       metric = metric, verbose = verbose,
                       random_state = random_state)
    
    return ks_random, n_random
