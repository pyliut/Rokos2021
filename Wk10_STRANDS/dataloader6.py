# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 13:45:54 2021

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

def dataloader6(ks_clusters, filename = "aaf_map.yaml"):
    """
    Convert clusters into a binary classification format

    Parameters
    ----------
    ks_clusters : pandas df
        columns = ["edge_id", "cluster_id"]
    filename : STR
        topological map file has contextual information. The default is "aaf_map.yaml".

    Returns
    -------
    df_class_diff : pandas df
        columns = ["edge1", "edge2", "same_cluster",
                   "edge_length_diff", "origin_connections_diff", 
                  "target_connections_diff", "total_connections_diff", 
                  "max_angle_diff", "sum_angle_diff"]
    

    """
    tic = time.time()
    #1) get context
    length = get_length(filename,suppress_message = True)
    connections = get_connections(filename,suppress_message = True)
    angle_max = get_angle_max(filename,suppress_message = True)
    angle_sum = get_angle_sum(filename,suppress_message = True)
    
    #2) Augment ks_clusters with context
    df_class = pd.DataFrame(columns = ["edge_id", "cluster_id", "edge_length", "n_connections_origin", "n_connections_target","n_connections_total", "max_angle","sum_angle"])
    df_class["edge_id"] = ks_clusters["edge_id"]
    df_class["cluster_id"] = ks_clusters["cluster_id"]
    for i in range(len(ks_clusters)):
        df_class["edge_length"][i] = length[df_class["edge_id"][i]]
        df_class["n_connections_origin"][i] = connections[df_class["edge_id"][i]][0]
        df_class["n_connections_target"][i] = connections[df_class["edge_id"][i]][1]
        df_class["n_connections_total"][i] = connections[df_class["edge_id"][i]][0] + connections[df_class["edge_id"][i]][1]
        df_class["max_angle"][i] = angle_max[df_class["edge_id"][i]]
        df_class["sum_angle"][i] = angle_sum[df_class["edge_id"][i]]
    
    #3) Turn into a binary classification problem
    # Output is binary - are 2 edges in the same cluster?
    # Inputs are the differences in spatial features
    max_n= int(np.ceil((len(df_class) * (len(df_class)-1)) / 2))
    df_class_diff = pd.DataFrame(index = np.arange(max_n), columns = ["edge1", "edge2", "same_cluster",
                                                                      "edge_length_diff", "origin_connections_diff", 
                                                                      "target_connections_diff", "total_connections_diff", 
                                                                      "max_angle_diff", "sum_angle_diff"])
    ind = 0
    for i in range(len(df_class)-1):
        for j in range(i+1, len(df_class)):
            df_class_diff["edge1"][ind] = df_class["edge_id"][i]
            df_class_diff["edge2"][ind] = df_class["edge_id"][j]
            if df_class["cluster_id"][i] == df_class["cluster_id"][j]:
                df_class_diff["same_cluster"][ind] = 1
            else:
                df_class_diff["same_cluster"][ind] = 0
            df_class_diff["edge_length_diff"][ind] = np.abs( df_class["edge_length"][i] - df_class["edge_length"][j])
            df_class_diff["origin_connections_diff"][ind] = np.abs( df_class["n_connections_origin"][i] - df_class["n_connections_origin"][j])
            df_class_diff["target_connections_diff"][ind] = np.abs( df_class["n_connections_target"][i] - df_class["n_connections_target"][j])
            df_class_diff["total_connections_diff"][ind] = np.abs( df_class["n_connections_total"][i] - df_class["n_connections_total"][j])
            df_class_diff["max_angle_diff"][ind] = np.abs( df_class["max_angle"][i] - df_class["max_angle"][j])
            df_class_diff["sum_angle_diff"][ind] = np.abs( df_class["sum_angle"][i] - df_class["sum_angle"][j])
            
            ind += 1
    
    toc = time.time()
    print("Time taken (dataloader6):", toc-tic, "secs")
    
    return df_class_diff