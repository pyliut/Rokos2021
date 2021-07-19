# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 14:49:18 2021

@author: pyliu
"""
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp

from underscore_prefix import *
from underscore_suffix import *

from select_data import *
from select_data_edge import *

def edge_plot(df, count, n_rows = 3, n_cols = 3):
    """
    Plots the data for the n_rows*n_cols edges with the most samples

    Parameters
    ----------
    df : Pandas DataFrame
        All data
    count : Pandas DataFrame
        ordered DataFrame of no. of samples per edge
    n_rows : INT
        no. of rows in the plot. The default is 3.
    n_cols : INT
        no. of cols in the plot. The default is 3.

    Returns
    -------
    STR
        Message to say that the process is completed.

    """
    #1) Check n_rows & n_cols is reasonable
    if n_cols > 10 or n_rows > 10:
        return "Max n_rows or n_cols is 10"
    
    #2) initialise subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_rows,5*n_cols))
    plt.xlabel("Duration (s)")
    plt.ylabel("Probability")
    
    #3) individual plots
    for i in range(n_rows):
        for j in range(n_cols):
            ind = i*n_rows + j
            
            #3a) Select waypoints
            edge = count["edge_id"][ind]
            #3b) Select edge in one direction
            subset = select_data_edge(df, edge)
            #3d) Duration data
            t_op = subset["operation_time"]
            
            # 4) subplots
            axs[i,j].hist(t_op, density = True, bins = 100);
            axs[i,j].set_title(edge)
            
    return "Done"
    
    