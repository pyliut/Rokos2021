# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 21:11:54 2021

@author: pyliu
"""

import pandas as pd
import numpy as np
import scipy as sp
import time

from get_length import *
from get_connections import *
from get_angle_max import *
from get_angle_sum import *
from select_data_edge import *



def similar_classification_fast(edge, df_train, clf, 
                           length_train, length_test, 
                           origin_train, origin_test,
                           target_train, target_test,
                           angle_max_train,angle_max_test,
                           angle_sum_train, angle_sum_test,
                           cutoff = 1):
    """
    Get most similar edge from training map by edge length with at least cutoff datapionts

    Parameters
    ----------
    edge : STR
        test edge name
    clf : OBJ
        pre-trained classifier
        features are ["edge_length_diff", "origin_connections_diff", 
                            "target_connections_diff", "total_connections_diff", 
                            "max_angle_diff", "sum_angle_diff"]


    Returns
    -------
    similar_edges : STR
        most similar edges from training map

    """
    
    
    #1) create pandas df to store context & predicted result
    edges_train = list(length_train.keys())
    df_predict = pd.DataFrame(index = np.arange(len(edges_train)), 
                             columns = ["edge_test","edge_train",
                                        "class_label","class0_prob","class1_prob",
                                       "edge_length_diff", "origin_connections_diff", 
                                      "target_connections_diff", "total_connections_diff", 
                                      "max_angle_diff", "sum_angle_diff",
                                      "length1", "length2",
                                      "origin1", "origin2", 
                                      "target1", "target2",
                                      "total1", "total2",
                                      "max1", "max2", 
                                      "sum1", "sum2"])
    df_predict["edge_test"] = edge
    df_predict["edge_train"] = edges_train
    
    df_predict["length1"] = edge
    df_predict["length2"] = edges_train
    df_predict["length1"] = df_predict["length1"].replace(to_replace = length_test)
    df_predict["length2"] = df_predict["length2"].replace(to_replace = length_train)
    
    df_predict["origin1"] = edge
    df_predict["origin2"] = edges_train
    df_predict["origin1"] = df_predict["origin1"].replace(to_replace = origin_test)
    df_predict["origin2"] = df_predict["origin2"].replace(to_replace = origin_train)
    
    df_predict["target1"] = edge
    df_predict["target2"] = edges_train
    df_predict["target1"] = df_predict["target1"].replace(to_replace = target_test)
    df_predict["target2"] = df_predict["target2"].replace(to_replace = target_train)
    
    df_predict["total1"] = df_predict["origin1"] + df_predict["target1"] 
    df_predict["total2"] = df_predict["origin2"] + df_predict["target2"] 
    
    df_predict["max1"] = edge
    df_predict["max2"] = edges_train
    df_predict["max1"] = df_predict["max1"].replace(to_replace = angle_max_test)
    df_predict["max2"] = df_predict["max2"].replace(to_replace = angle_max_train)
    
    df_predict["sum1"] = edge
    df_predict["sum2"] = edges_train
    df_predict["sum1"] = df_predict["sum1"].replace(to_replace = angle_sum_test)
    df_predict["sum2"] = df_predict["sum2"].replace(to_replace = angle_sum_train)
    
    #2) Calculate all feature differences
    df_predict["edge_length_diff"] = np.abs( df_predict["length1"] - df_predict["length2"] )
    df_predict["origin_connections_diff"] = np.abs( df_predict["origin1"] - df_predict["origin2"] )
    df_predict["target_connections_diff"] = np.abs( df_predict["target1"] - df_predict["target2"] )
    df_predict["total_connections_diff"] = np.abs( df_predict["total1"] - df_predict["total2"] )
    df_predict["max_angle_diff"] = np.abs( df_predict["max1"] - df_predict["max2"])
    df_predict["sum_angle_diff"] = np.abs( df_predict["sum1"] - df_predict["sum2"])
    
    #3) predict outputs
    X = np.array(df_predict[["edge_length_diff", "origin_connections_diff", 
                            "target_connections_diff", "total_connections_diff", 
                            "max_angle_diff", "sum_angle_diff"]])
    df_predict["class_label"] = clf.predict(X)
    y_prob =  clf.predict_proba(X)
    df_predict["class0_prob"] = y_prob[:,0]
    df_predict["class1_prob"] = y_prob[:,1]
    
    #4) Sort array so that top values are most similar (lowest class0_prob)
    df_predict = df_predict.sort_values("class0_prob", ascending = True).reset_index(drop=True)
    index = 0
    edge_train = df_predict["edge_train"][index]
    subset_train = select_data_edge(df_train,edge_train)
    while len(subset_train) < cutoff:
        index += 1
        if index > len(df_predict):
            raise ValueError("Error: no data")
        edge_train = df_predict["edge_train"][index]
        subset_train = select_data_edge(df_train,edge_train)
    
    return edge_train