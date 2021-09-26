# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 12:26:47 2021

@author: pyliu
"""

import pandas as pd
import numpy as np
import scipy as sp
import time

from underscore_prefix import *
from underscore_suffix import *
from select_data_edge import *
from get_length import *
from get_connections import *
from get_angle_max import *
from get_angle_sum import *
from error_square_2samples import *

def regression_dataloader_square(df, filename = "aaf_map.yaml",metric = "difference",cutoff = 1, verbose = False, report_interval = 5000):
    """
    Get KS scores & spatial features between edges
    This one does not calculate KS score for fitted lognormal distributions

    Parameters
    ----------
    df : pd df
        observations
    fit : pandas df
        contains fitted params for Bayesian lognormal optimisation
    filename : STR
        name of YAML file containing contextual information. The default is "aaf_map.yaml".
    metric : STR
        Valid metrics are "difference" & "operation_time". The default is "difference".
    cutoff : INT
        Disregards edges with fewer observations than cutoff
    precision : INT
        precision of KS test for fitted distributions. The default is 2.
    verbose : BOOL
        If true, print statements to update user about progress. The default is False.
    report_interval : INT
        How frequently to update user. The default is 5000.

    Raises
    ------
    ValueError
        invalid metric.

    Returns
    -------
    df_ks_diff : pandas df
        columns = ["edge1","edge2",
                 "n_obs1","n_obs2",
                 "ks","square",
                "edge_length_diff", "origin_connections_diff", 
                "target_connections_diff", "total_connections_diff", 
                "max_angle_diff", "sum_angle_diff"]
        ks_raw is output of 2 sample KS test

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
    
    #2) initialise empty dataframe to store results
    n_edges = len(count)
    n_edge_pairs = int(n_edges * (n_edges-1)//2)
    df_ks_diff = pd.DataFrame(index = np.arange(n_edge_pairs), 
                              columns = ["edge1","edge2",
                                         "n_obs1","n_obs2",
                                         "ks","square",
                                        "edge_length_diff", "origin_connections_diff", 
                                        "target_connections_diff", "total_connections_diff", 
                                        "max_angle_diff", "sum_angle_diff"])

    #3) get context
    length = get_length(filename,suppress_message = True)
    connections = get_connections(filename,suppress_message = True)
    angle_max = get_angle_max(filename,suppress_message = True)
    angle_sum = get_angle_sum(filename,suppress_message = True)
    
    #4) Load fit data
    for i in range(n_edges-1):
        #load edge 1 data
        edge1 = count["edge_id"][i]
        subset1 = select_data_edge(df, edge1)
        #skip edge if below cutoff
        if len(subset1) < cutoff:
            continue
        
        #fit edge 1
        if metric == "operation_time":
            t_op1 = subset1["operation_time"]
        elif metric == "difference":
            t_op1 = subset1["operation_time"] - subset1["time_to_waypoint"]
        else:
            raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")
            
        
        for j in range(i+1,n_edges):
            index = i*n_edges + j
            
            #progress update
            if verbose == True:
                if index % report_interval == 0:
                    toc = time.time()
                    print(index, "iterations:", toc-tic, "secs")
            
            #load edge 2 data
            edge2 = count["edge_id"][j]
            subset2 = select_data_edge(df, edge2)
            #skip edge if below cutoff
            if len(subset2) < cutoff:
                continue

            #fit edge 2
            if metric == "operation_time":
                t_op2 = subset2["operation_time"]
            elif metric == "difference":
                t_op2 = subset2["operation_time"] - subset2["time_to_waypoint"]
            else:
                raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")
            
            #5) Calculate values for dataframe
            #5a) Edge names & no. of observations
            df_ks_diff["edge1"][index] = edge1
            df_ks_diff["edge2"][index] = edge2
            df_ks_diff["n_obs1"][index] = len(subset1)
            df_ks_diff["n_obs2"][index] = len(subset2)
            
            #5b) KS values
            ks_raw,p_val_raw = sp.stats.ks_2samp(t_op1,t_op2)
            df_ks_diff["ks"][index] = ks_raw
            
            area = error_square_2samples(t_op1, t_op2, plot_graph = False)
            df_ks_diff["square"][index] = area
            
            #5c) Spatial features
            df_ks_diff["edge_length_diff"][index] = np.abs(length[edge1] - length[edge2])
            df_ks_diff["origin_connections_diff"][index] = np.abs(connections[edge1][0] - connections[edge2][0])
            df_ks_diff["target_connections_diff"][index] = np.abs(connections[edge1][1] - connections[edge2][1])
            df_ks_diff["total_connections_diff"][index] = np.abs(np.sum(connections[edge1]) - np.sum(connections[edge2]))
            df_ks_diff["max_angle_diff"][index] = np.abs(angle_max[edge1] - angle_max[edge2])
            df_ks_diff["sum_angle_diff"][index] = np.abs(angle_sum[edge1] - angle_sum[edge2])
            
    
    df_ks_diff = df_ks_diff.dropna().reset_index(drop=True)
    toc = time.time()
    print("Time taken (regression_dataloader):", toc-tic,"secs")
    
    return df_ks_diff
            
    
    