# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:29:51 2021

@author: pyliu
"""
import numpy as np
import pandas as pd
import time

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

def kde(t_test, t_op, kernel = "gaussian", max_bandwidth = 5, bandwidth_precision = 0.1, cv = 20):
    
    tic = time.time()
    
    #1) select bandwidth
    grid = GridSearchCV(KernelDensity(kernel=kernel),
                    {'bandwidth': np.arange(bandwidth_precision, max_bandwidth, bandwidth_precision)},
                    cv=cv) # 20-fold cross-validation
    grid.fit(t_op[:, None])
    
    #fit KDE
    #kernel types: {‘gaussian’, ‘tophat’, ‘epanechnikov’, ‘exponential’, ‘linear’, ‘cosine’}, 
    #default=’gaussian’
    kde = KernelDensity(bandwidth=grid.best_params_["bandwidth"], kernel=kernel)
    kde.fit(t_op[:, np.newaxis])
    logprob = kde.score_samples(t_test[:,None])
    
    toc = time.time()
    print("Time taken (kde):", toc-tic, "secs")
    
    return np.exp(logprob)