# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 10:24:49 2021

@author: pyliu
"""
import pandas as pd
import numpy as np
import scipy as sp
import time
def distances_class(df_test,clf):
    """
    Converts class probabilities into a distance matrix for HAC clustering

    Parameters
    ----------
    df_test : pandas df
        Test set
    clf : object
        Trained classifier
        
    Returns
    -------
    dist_matrix : ARRAY of FLOAT
        distances
    dist_edges : ARRAY of STR
        corresponding edge names

    """
    tic = time.time()
    #1) use classifier to predict probabilities
    X_test = np.array(df_test[["edge_length_diff", "origin_connections_diff", 
                                  "target_connections_diff", "total_connections_diff", 
                                  "max_angle_diff", "sum_angle_diff"]])
    
    #y_prob[:,0] are probabilities of class 0
    #y_prob[:,1] are probabilities of class 1
    y_prob = clf.predict_proba(X_test) 
    y_dist = y_prob[:,0]
    
    #2) augment dataframe with predictions
    df_test["prob_pred"] = np.NaN
    df_test["dist_pred"] = np.NaN
    df_test["prob_pred"] = y_prob
    df_test["dist_pred"] = y_dist
    
    #3) Create distance matrix
    y_edge1 = df_test["edge1"]
    y_edge2 = df_test["edge2"]
    dist_edges = df_test["edge1"].unique()
    
    dist_matrix = np.empty( (len(df_test["edge1"].unique()), len(df_test["edge1"].unique())) )
    dist_matrix[:] = np.NaN
    
    index = 0
    for i in range(len(dist_edges)-1):
        for j in range(i,len(dist_edges)):
            if index % 5000 == 0:
                toc = time.time()
                print(index,"iterations:",toc-tic,"secs")
            if i == j:
                dist_matrix[i][j] = 0.0
            else:
                criterion1 = (df_test["edge1"]==dist_edges[i]) & (df_test["edge2"]==dist_edges[j])
                criterion2 = (df_test["edge2"]==dist_edges[i]) & (df_test["edge1"]==dist_edges[j])
                criterion = (criterion1 | criterion2)
                dist_matrix[i][j] = float(df_test.loc[criterion]["dist_pred"])
                dist_matrix[j][i] = float(df_test.loc[criterion]["dist_pred"])
            index += 1
    dist_matrix[len(dist_edges)-1][len(dist_edges)-1] = 0.0
    
    toc = time.time()
    print("Time taken (distances_class):",toc-tic,"secs")
    
    return dist_matrix, dist_edges
    
    
    
    
    
    
    
    
    