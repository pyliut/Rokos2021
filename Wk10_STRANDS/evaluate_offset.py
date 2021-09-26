# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 15:03:13 2021

@author: pyliu
"""
import yaml
import pandas as pd
import numpy as np
import scipy as sp
import time

from get_length import *
from select_data_edge import *

def evaluate_offset(df, filename = "aaf_map.yaml", metric = "difference", cutoff = 20):
    """
    Get the maximum speeds on all edges in a map

    Parameters
    ----------
    df : pandas df
        observations
    filename : STR
        name of topological map. The default is "aaf_map.yaml".
    metric : STR
        Valid metrics are ["operation_time","difference"]. The default is "difference".
    cutoff : INT
        Minimum number of datapoints to perform fitting.  The default is 20.

    Raises
    ------
    ValueError
        Invalid metric

    Returns
    -------
    df_offset : pandas df
        columns = ["edge_id", "edge_length", "t_min", "max_speed"]

    """
    
    length = get_length(filename, suppress_message = True)
    edges = list( length.keys() )
    df_offset = pd.DataFrame(index = np.arange(len(edges)),
                             columns = ["edge_id", "edge_length", "t_min", "max_speed"])
    
    for i, edge in enumerate(edges):
        subset = select_data_edge(df,edge)
        if metric == "operation_time":
            t_op = subset["operation_time"]
        elif metric == "difference":
            t_op = subset["operation_time"] - subset["time_to_waypoint"]
        else:
            raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")
        
        if len(subset) >= cutoff:
            df_offset["edge_id"][i] = edge
            df_offset["edge_length"][i] = length[edge]
            df_offset["t_min"][i] = np.min(t_op)
            df_offset["max_speed"][i] = df_offset["edge_length"][i]/df_offset["t_min"][i]
    
    print("Max speed:",df_offset["max_speed"].max())
    print("Mean:",df_offset["max_speed"].mean())
    print("Std:",df_offset["max_speed"].std())
    
    return df_offset