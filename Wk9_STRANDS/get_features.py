# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 16:53:56 2021

@author: pyliu
"""
import pandas as pd
import scipy as sp
import numpy as np
import time

from get_length import *
from get_connections import *
from get_angle_max import *
from get_angle_sum import *


def get_features(filename = "aaf_map.yaml"):
    """
    Get features of all edges in map

    Parameters
    ----------
    filename : STR
        name of YAML file containing contextual information. The default is "aaf_map.yaml".

    Returns
    -------
    df_features : pandas df
        columns = ["edge_id","edge_length","sum_angle","max_angle",
                 "origin_connections","target_connections","total_connections"])
    

    """
    tic = time.time()
    #1) Get context
    length = get_length(filename,suppress_message = True)
    connections = get_connections(filename,suppress_message = True)
    angle_max = get_angle_max(filename,suppress_message = True)
    angle_sum = get_angle_sum(filename,suppress_message = True)
    
    #2) Create df to store features
    edges = list(length.keys())
    
    df_features = pd.DataFrame(index = np.arange(len(edges)),
                               columns = ["edge_id","edge_length","sum_angle","max_angle",
                                          "origin_connections","target_connections","total_connections"])
    for i,edge in enumerate(edges):
        df_features["edge_id"][i] = edge
        df_features["edge_length"][i] = length[edge]
        df_features["sum_angle"][i] = angle_sum[edge]
        df_features["max_angle"][i] = angle_max[edge]
        df_features["origin_connections"][i] = connections[edge][0]
        df_features["target_connections"][i] = connections[edge][0]
        df_features["total_connections"][i] = np.sum(connections[edge])
        
    toc = time.time()
    print("Time taken (get_features):",toc-tic,"secs")
    
    return df_features