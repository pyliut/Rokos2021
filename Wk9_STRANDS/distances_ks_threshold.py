# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 12:22:37 2021

@author: pyliu
"""

import pandas as pd
import numpy as np
import math
import scipy as sp
import time

from select_data_edge import *
from underscore_prefix import *
from underscore_suffix import *

def distances_ks_threshold(df, metric = "operation_time", cutoff = 1, threshold = 0.9):
    """
    Returns difference matrix between edges and a list of the edge names
    Difference metric is KS statistic

    Parameters
    ----------
    context : Pandas DataFrame
        Columns are "edge_id", "origin", "target", "edge_length", "n_connections_origin","n_connections_target"
    metric : STR
        valid metrics = ["operation_time", "difference"]
    cutoff : INT
        Disregards edges with fewer observations than cutoff
    threshold : FLOAT
        Removes edges with KS > threshold for all other edges
        
    Returns
    -------
    diff_matrix : 2D array
        Difference metric between edges
    edge_list : array
        List of edge_id

    """
    tic = time.time()
    #turn off warnings
    pd.options.mode.chained_assignment = None  # default='warn'
    
    valid_metrics = ["operation_time", "difference"]
    if metric not in valid_metrics:
        raise ValueError("Invalid metric. Valid metrics are ['operation_time', 'difference']")
    
    #1) Order edges by amount of data
    #1a) Order the data in terms of edges with the largest no. of observations
    count = df["edge_id"].value_counts()        #sort by no. of samples per edge
    count.to_csv('waypoint_pairs.csv')          #save and reload for DataFrame format
    count = pd.read_csv("waypoint_pairs.csv")
    count.columns = ["edge_id", "samples"]      #rename columns
    
    #1b) split edge_id into target & origin
    count["origin"] = None          #Add new columns
    count["target"] = None
    
    for i in range(len(count["edge_id"])):
        count["origin"][i] = underscore_prefix(str(count["edge_id"][i]))
        count["target"][i] = underscore_suffix(str(count["edge_id"][i]))
    
    #2) initialise empty matrix to store results
    ks_matrix = []
    edge_list = []
    n_edge = len(count)
    invalid_indices = []
    lone_edges = []
    
    #3) calculate distances
    for i in range(n_edge):
        if i % 10 == 0:
            toc = time.time()
            print(i, "edges:", toc-tic, "secs")
            
        ks_list = []
        #3a) Select first edge
        wp1 = count["origin"][i]
        wp2 = count["target"][i]
        edge1 = str(wp1) + "_" + str(wp2)
        subset1 = select_data_edge(df, edge1)
        if len(subset1) < cutoff:
            continue
        if metric == "operation_time":
            t_op1 = subset1["operation_time"]
        else:
            t_op1 = subset1["operation_time"] - subset1["time_to_waypoint"]
        edge_list.append(edge1)
        
        for j in range(n_edge):
            #3b) Select second edge
            wp1 = count["origin"][j]
            wp2 = count["target"][j]
            edge2 = str(wp1) + "_" + str(wp2)
            subset2 = select_data_edge(df, edge2)
            if len(subset2) < cutoff:
                continue
            if metric == "operation_time":
                t_op2 = subset2["operation_time"]
            else:
                t_op2 = subset2["operation_time"] - subset2["time_to_waypoint"]
            
            #calculate ks statistic & p-value between edges
            ks_stat, p_val = sp.stats.ks_2samp(t_op1,t_op2)
            
            ks_list.append(ks_stat)
        ks_matrix.append(ks_list)
        
        #check whether above threshold
        check_list = np.delete(ks_list, i)
        if np.min( check_list ) > threshold:
            invalid_indices.append(i)
            lone_edges.append(edge1)
    
    if len(invalid_indices) > 0:
        ks_matrix = np.delete(ks_matrix, invalid_indices, axis = 0)
        ks_matrix = np.delete(ks_matrix, invalid_indices, axis = 1)
        edge_list = np.delete(edge_list, invalid_indices)
        
    
    toc = time.time()
    print("Time taken:", toc-tic, "secs")
    
    return ks_matrix, edge_list, lone_edges
            
    