# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 12:21:10 2021

@author: pyliu
"""
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from select_data_edge import *

def compare_edges(df,edge1,edge2,metric = "difference", precision = 2, n_bins = 50, t_max = 30):
    """
    Compare KS score of 2 different edges

    Parameters
    ----------
    df : pandas df
        observations
    edge1 : STR
        name of first edge
    edge2 : STR
        name of second edge
    metric : STR
        Options are "difference" and "operation_time". The default is "difference".
    precision : INT
        spacing of time axis. The default is 2.
    n_bins : INT
        number of bins in pdf histogram. The default is 50.
    t_max : FLOAT
        max value on time axis. The default is 30.

    Raises
    ------
    ValueError
        metric is invalid

    Returns
    -------
    ks : FLOAT
        KS value
    p_val : FLOAT
        p-value

    """
    # 1) Select data of interest
    origin = "WayPoint91"
    target = "WayPoint86"
    edge1 = origin + "_" + target
    subset1 = select_data_edge(df, edge1)
    #independent variable to plot over
    if metric == "difference":
        t_op1 = subset1["operation_time"] - subset1["time_to_waypoint"] 
    elif metric == "operation_time":
        t_op1 = subset1["operation_time"] 
    else:
        raise ValueError("Invalid metric. Valid metrics are ['difference', 'operation_time']")
    
    origin = "WayPoint19"
    target = "WayPoint21"
    edge2 = origin + "_" + target
    subset2 = select_data_edge(df, edge2)
    #independent variable to plot over
    if metric == "difference":
        t_op2 = subset2["operation_time"] - subset2["time_to_waypoint"] 
    elif metric == "operation_time":
        t_op2 = subset2["operation_time"] 
    else:
        raise ValueError("Invalid metric. Valid metrics are ['difference', 'operation_time']")
    
    #Plot against actual data
    t_start= 10**(-precision)
    t_stop = ( (np.max(t_op1)) //5)*5 + 5    #round up to nearest 5 secs
    t_step = 10**(-precision)
    t_test = np.arange(t_start,t_stop,t_step)
    
    
    #plot actual
    print("N_observations:", len(subset1), len(subset2))
    ks,p_val = sp.stats.ks_2samp(t_op1, t_op2)
    print("KS stat:", ks, "    p_val:", p_val)
    plt.hist(t_op1, density = True, bins = n_bins, alpha = 0.5);
    plt.hist(t_op2, density = True, bins = n_bins, alpha = 0.5);
    
    plt.hist(t_op1, bins = 2000, density=True, histtype='step',cumulative=True)
    plt.hist(t_op2, bins = 2000, density=True, histtype='step',cumulative=True)
    
    plt.legend([edge1, edge2, edge1, edge2])
    plt.xlabel("operation_time (s)")
    plt.ylabel("probability")
    plt.xlim([0,np.min([t_max,t_stop])])
    plt.title(edge1 + " vs " + edge2)
    return ks, p_val