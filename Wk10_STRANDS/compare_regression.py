# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 17:37:31 2021

@author: pyliu
"""

import pandas as pd
import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt

from select_data_edge import *
from similar_regression import *
from compare_similar import *

def compare_regression(edge, clf, df_test, df_train, 
                   filename_test= "aaf_map.yaml",filename_train = "tsc_map.yaml", 
                   n_similar=5,cutoff = 10,
                   metric = "difference", x_max = 30, n_bins_test = 100, n_bins_train = 100):
    

    similar_edges, df_predict = similar_regression(edge, clf = clf, 
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
    print("predicted ks:", df_predict["ks"][index])
    print("length diff:", df_predict["edge_length_diff"][index])
    print("n_obs:", len(subset))
    
    ks = compare_similar(edge_test = edge, df_test = df_test, 
                    edge_train = edge_prior, df_train = df_train, 
                    metric = metric, x_max = x_max,
                    n_bins_test = n_bins_test, n_bins_train = n_bins_train)
    
    return ks