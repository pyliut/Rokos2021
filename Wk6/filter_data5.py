# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:54:59 2021

@author: pyliu
"""

import pandas as pd
from datetime import datetime
from datetime import timedelta
import time

def filter_data5(df, adjacent):
    """
    Add a column to the dataframe to show how many robots (in total) are on an edge at any time
    This is an alternative to filter_data2.py
    Difference is that filter_data2.py only considers congestion if 2 robots are on the same edge
    filter_data4.py also considers cases where robots are on adjacent edges

    Parameters
    ----------
    df : Pandas DataFrame
        Initial DataFrame 
    
    adjacent : DICT
        Key is a node (STR)
        Value is list of adjacent nodes

    Returns
    -------
    df : Pandas DataFrame
        Altered DataFrame 

    """
    tic = time.time()
    pd.options.mode.chained_assignment = None  # default='warn'

    df["n_robots"] = 0
    
    for i, robots in enumerate(df["n_robots"]):
        if i % 1000 == 0:
            toc = time.time()
            print(i,"iterations", toc-tic, "secs")
        edge_list = []
        target = df["target"][i]
        origin = df["origin"][i]
        edge_list.append(target + "_" + origin)
        
        #only consider edges adjacent to the target node   
        for wp in adjacent[target]:
            edge_list.append(wp + "_" + target)
        
        subset = df.loc[df["edge_id"].isin(edge_list),:]
        for j, t_start in enumerate(subset["start"]):
            if t_start >= df["start"][i] and t_start <= df["finish"][i]:
                df["n_robots"][i] += 1
    
    toc = time.time()
    print("Time taken", toc-tic, "secs")
    return df