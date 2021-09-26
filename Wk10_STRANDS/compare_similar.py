# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 20:23:21 2021

@author: pyliu
"""
import pandas as pd
import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt

from select_data_edge import *

def compare_similar(edge_test, df_test, edge_train, df_train, 
                    metric = "difference", x_max = 30,
                    n_bins_test = 100, n_bins_train = 100):
    
    #1) get data
    subset_test = select_data_edge(df_test,edge_test)
    subset_train = select_data_edge(df_train,edge_train)
    if metric == "operation_time":
        t_op_test = subset_test["operation_time"]
        t_op_train = subset_train["operation_time"]
    elif metric == "difference":
        t_op_test = subset_test["operation_time"] - subset_test["time_to_waypoint"]
        t_op_train = subset_train["operation_time"] - subset_train["time_to_waypoint"]
    else:
        raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")
    
    print("n_test:",len(subset_test))
    print("n_train:",len(subset_train))
    #2) Plot histograms
    plt.hist(t_op_test, density = True, bins = n_bins_test, alpha = 0.5, label = "test: " + edge_test);
    plt.hist(t_op_train, density = True, bins = n_bins_train, alpha = 0.5, label = "train: " + edge_train);
    
    plt.hist(t_op_test, bins = 2000, density=True, 
             linestyle = "dashed",histtype='step',cumulative=True, label = "test: " + edge_test)
    plt.hist(t_op_train, bins = 2000, density=True, 
             linestyle = "dashed",histtype='step',cumulative=True, label = "train: " + edge_train)
    
    plt.xlim([0,x_max])
    plt.ylim([0,1])
    plt.xlabel("duration (s)")
    plt.ylabel("probability")
    plt.legend(bbox_to_anchor=(1, 1.05), loc='upper left')
    
    ks,pval = sp.stats.ks_2samp(t_op_test,t_op_train)
    print("KS:", ks)
    return ks