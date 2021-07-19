# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 13:18:27 2021

@author: pyliu
"""
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

from underscore_prefix import *
from underscore_suffix import *

from select_data import *
from select_data_edge import *


def edge_direction(df, n_plots = 5, p_crit = 0.05):
    """
    Plots forward and back transitions on n_plots edges with most data
    returns an ordered DataFrame of no. of samples per edge (order matters)

    Parameters
    ----------
    df : Pandas DataFrame
        All data
    n_plots : INT
        No. of plots to make. The default is 5.
    p_crit : FLOAT between 0 & 1
        Critical p-value for ks test. The default is 0.05.

    Returns
    -------
    count : Pandas DataFrame
        ordered DataFrame of no. of samples per edge

    """
    #check that p_crit is valid (between 0 & 1)
    if p_crit > 1 or p_crit < 0:
        return "ERROR: p_crit must be between 0 & 1"
    
    
    #1) Get data into correct format
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
    
    #1c) create 2 new columns to calculate Kolmogorov-smirnov statistic
    count["ks statistic"] = None
    count["ks p-value"] = None
    
    #1d) add a column to store whether the edge can be swapped
    count["reversible"] = False
    
    
    #2a) initialise subplots
    fig, axs = plt.subplots(n_plots, figsize=(10,5*n_plots))
    plt.xlabel("Duration (s)")
    plt.ylabel("Probability")
    
    #3) plot
    for i in range(len(count["edge_id"])):
        #3a) Select waypoints
        wp1 = count["origin"][i]
        wp2 = count["target"][i]
    
        #3b) Select edge in one direction
        edge1 = str(wp1) + "_" + str(wp2)
        subset1 = select_data_edge(df, edge1)
    
        #3c) Select edge in other direction
        edge2 = str(wp2) + "_" + str(wp1)
        subset2 = select_data_edge(df, edge2)
    
        #3d) Duration data
        t_op1 = subset1["operation_time"]
        t_op2 = subset2["operation_time"]
        
        # 4) subplots
        if i < n_plots:
            axs[i].hist(t_op1, density = True, bins = 100);
            axs[i].hist(t_op2, density = True, bins = 100, alpha = 0.5);
            axs[i].set_title(str(wp1) + " & " + str(wp2))
        
        #5) Calculate k-s statistic and p-value
        if len(t_op1) > 0 and len(t_op2) > 0:
            count["ks statistic"][i], count["ks p-value"][i] = sp.stats.kstest(t_op1, t_op2)
            if count["ks p-value"][i] >= p_crit:
                count["reversible"][i] = True

    return count