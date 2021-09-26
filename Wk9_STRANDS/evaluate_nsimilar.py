# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 20:21:50 2021

@author: pyliu
"""
import scipy as sp
import pandas as pd
import numpy as np
import time

from select_data_edge import *

def evaluate_nsimilar(df_similar,df_seen,df_unseen,cutoff = 20,metric = "difference"):
    
    
    tic = time.time()
    n_similar = len(df_similar["similar_edge"][0])
    n_total = 0
    mse = 0
    mae = 0
    mean_ks = 0
    for i in range(len(df_similar)):
        if i % 25 == 0:
            toc = time.time()
            print(i,"edges:",toc-tic,"secs")
        
        
        edge1 = df_similar["edge_id"][i]
        subset1 = select_data_edge(df_unseen, edge1)
        #fit edge 1
        if metric == "operation_time":
            t_op1 = subset1["operation_time"]
        elif metric == "difference":
            t_op1 = subset1["operation_time"] - subset1["time_to_waypoint"]
        else:
            raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")
        
        if len(subset1) < cutoff:
            continue
            
        for j in range(n_similar):
            edge2 = df_similar["similar_edge"][i][j]
            ks_pred = df_similar["similar_ks"][i][j]
            subset2 = select_data_edge(df_seen, edge2)
            #fit edge 2
            if metric == "operation_time":
                t_op2 = subset2["operation_time"]
            elif metric == "difference":
                t_op2 = subset2["operation_time"] - subset2["time_to_waypoint"]
            else:
                raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")
                
            if len(subset2) < cutoff:
                continue
            
            ks_real,pval_real = sp.stats.ks_2samp(t_op1,t_op2)
            error = np.abs(ks_real-ks_pred)
            mean_ks += ks_real
            mae += error
            mse += error*error
            n_total += 1
    
    mse /= n_total
    mae /= n_total
    mean_ks /= n_total

    toc = time.time()
    print("Time taken (evaluate_nsimilar):",toc-tic,"secs")
    print("Mean KS:",mean_ks)
    print("MAE:",mae)
    print("MSE:",mse)
    
    return mean_ks,mae,mse
            