# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 17:54:00 2021

@author: pyliu
"""


import scipy as sp
import numpy as np
import pandas as pd
import random

import time

from evaluate_random import *
from random_prior import *
from specific_prior import *
from update_mean import *
from update_var import *

def evaluate_random_multiedge(edge,df_test, df_train, filename_train = "tsc_map.yaml",
                    metric = "difference", cutoff=100, max_obs = 10,n_repeats = 5,
                    plot_graph = False, verbose = True,
                    random_state = None):
    """
    Ensemble wrapper for evaluate_random.py, random_prior.py, specific_prior.py

    Additional Parameters
    ----------
    n_repeats : INT
        number of iterations in ensemble. The default is 5.


    """
    
    tic = time.time()

    #create seeds
    random.seed(random_state)
    seeds = [random.randint(0,1024) for i in range(n_repeats)]
    
    #initial run
    mean_test, var_test, prior, t_op_prior, edge_prior = random_prior(df_train,
                                            filename = filename_train,metric = metric, 
                                            cutoff=cutoff, max_obs = max_obs, 
                                            plot_graph = plot_graph,
                                            random_state = seeds[0])

    ks_list, n_list = evaluate_prior(edge, df_test, 
                       mean_test, var_test, prior, t_op_prior,
                       metric = metric, verbose = verbose,
                       random_state = seeds[0])
    ks_list = np.array(ks_list)
    n_list = np.array(n_list)
    std_list = update_var(ks_list,0,0,ks_list)
    
    #additional runs
    for i in range(1,n_repeats):
        if (i+1) % 2 == 0 and verbose == True:
            toc = time.time()
            print(i+1,"iterations:",toc-tic,"secs")
            
        mean_test, var_test, prior, t_op_prior, edge_prior = random_prior(df_train,
                                            filename = filename_train,metric = metric, 
                                            cutoff=cutoff, max_obs = max_obs, 
                                            plot_graph = plot_graph,
                                            random_state = seeds[i])
    
        ks_current, n_current = evaluate_prior(edge, df_test, 
                       mean_test, var_test, prior, t_op_prior,
                       metric = metric, verbose = verbose,
                       random_state = seeds[i])
        ks_list = update_mean(ks_list,i,ks_current)
        std_list = update_var(ks_list,std_list,i,ks_current)
        
    #find std
    std_list = np.sqrt(std_list)
    
    toc = time.time()
    print("Time taken (evaluate_random_ensemble):",toc-tic,"secs")
    return ks_list,n_list,std_list