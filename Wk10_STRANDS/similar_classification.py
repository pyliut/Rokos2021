# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 14:38:25 2021

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



def similar_classification(edge, clf, 
                           filename_test = "aaf_map.yaml", filename_train = "tsc_map.yaml", 
                           n_similar = 1):
    """
    Get n_similar most similar edges from training map by edge length
    Also returns a dataframe with more context

    Parameters
    ----------
    edge : STR
        test edge name
    clf : OBJ
        pre-trained classifier
        features are ["edge_length_diff", "origin_connections_diff", 
                            "target_connections_diff", "total_connections_diff", 
                            "max_angle_diff", "sum_angle_diff"]
        
    filename_test : STR
        Name of file containing topological map data for test map. The default is "aaf_map.yaml".
    filename_train : STR
        Name of file containing topological map data for train map. The default is "tsc_map.yaml".
    n_similar : INT
        number of similar edges returned. The default is 1.

    Returns
    -------
    similar_edges : STR, vector
        list of n_similar most similar edges from training map
    df_predict : pandas df
        columns are ["edge_test","edge_train",
                     "class_label","class0_prob","class1_prob",
                     "edge_length_diff", "origin_connections_diff", 
                     "target_connections_diff", "total_connections_diff", 
                     "max_angle_diff", "sum_angle_diff"]

    """
    
    #1) get context
    length_test = get_length(filename_test, suppress_message=True)
    length_train = get_length(filename_train, suppress_message=True)
    connections_test = get_connections(filename_test, suppress_message=True)
    connections_train = get_connections(filename_train, suppress_message=True)
    angle_max_test = get_angle_max(filename_test, suppress_message=True)
    angle_max_train = get_angle_max(filename_train, suppress_message=True)
    angle_sum_test = get_angle_sum(filename_test, suppress_message=True)
    angle_sum_train = get_angle_sum(filename_train, suppress_message=True)
    
    #2) create pandas df to store context & predicted result
    edges_train = list(length_train.keys())
    df_predict = pd.DataFrame(index = np.arange(len(edges_train)), 
                             columns = ["edge_test","edge_train",
                                        "class_label","class0_prob","class1_prob",
                                       "edge_length_diff", "origin_connections_diff", 
                                      "target_connections_diff", "total_connections_diff", 
                                      "max_angle_diff", "sum_angle_diff"])
    df_predict["edge_test"] = edge
    df_predict["edge_train"] = edges_train
    for i in range(len(edges_train)):
        df_predict["edge_length_diff"][i] = np.abs(length_test[df_predict["edge_test"][i]] - length_train[df_predict["edge_train"][i]])
        df_predict["origin_connections_diff"][i] = np.abs(connections_test[df_predict["edge_test"][i]][0] - connections_train[df_predict["edge_train"][i]][0])
        df_predict["target_connections_diff"][i] = np.abs(connections_test[df_predict["edge_test"][i]][1] - connections_train[df_predict["edge_train"][i]][1])
        df_predict["total_connections_diff"][i] = np.abs(np.sum(connections_test[df_predict["edge_test"][i]]) - np.sum(connections_train[df_predict["edge_train"][i]]))
        df_predict["max_angle_diff"][i] = np.abs(angle_max_test[df_predict["edge_test"][i]] - angle_max_train[df_predict["edge_train"][i]])
        df_predict["sum_angle_diff"][i] = np.abs(angle_sum_test[df_predict["edge_test"][i]] - angle_sum_train[df_predict["edge_train"][i]])
    
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
    similar_edges = list(df_predict["edge_train"][:n_similar])
    
    return similar_edges, df_predict