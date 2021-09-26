# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:22:05 2021

@author: pyliu
"""
import scipy as sp
import numpy as np
import pandas as pd
import random

import time

from evaluate_mle import *
from update_mean import *
from update_var import *

def evaluate_mle_ensemble(edge,df, metric = "difference", n_repeats = 5,
                          verbose = False, random_state = None):
    """
    Ensemble wrapper for evaluate_mle.py

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
    ks_list, n_list = evaluate_mle(edge,df, metric = metric, 
                                   verbose = False, random_state = seeds[0])
    ks_list = np.array(ks_list)
    n_list = np.array(n_list)
    std_list = update_var(ks_list,0,0,ks_list)
    
    #additional runs
    for i in range(1,n_repeats):
        if (i+1) % 5 == 0 and verbose == True:
            toc = time.time()
            print(i+1,"iterations:",toc-tic,"secs")
        ks_current, n_current = evaluate_mle(edge,df, metric = metric, 
                                             verbose = False, random_state = seeds[i])
        ks_list = update_mean(ks_list,i,ks_current)
        std_list = update_var(ks_list,std_list,i,ks_current)
        
    #find std
    std_list = np.sqrt(std_list)
    
    print("Time taken (evaluate_mle_ensemble):",toc-tic,"secs")
    return ks_list,n_list,std_list