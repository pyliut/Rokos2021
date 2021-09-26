# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:32:43 2021

@author: pyliu
"""

import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
import math
import time

from underscore_prefix import *
from underscore_suffix import *

def get_congestion(df, adjacent):
    """
    Use after get_data & get_adjacent
    Use before get_valid

    Parameters
    ----------
    df : pandas df
        columns = ["_id", "origin", "target", "succeeded", "run_id", "date_finished",
                   "policy_goal", "agent", "_meta", "is_final", "edge_id", "action", 
                   "date_started", "operation_time", "topological_map", "final_node"]
    adajacent : DICT
        key is a node
        Values are nodes which are adjacent to key node

    Returns
    -------
    df : pandas df
        columns = ["_id", "origin", "target", "succeeded", "run_id", "date_finished",
                   "policy_goal", "agent", "_meta", "is_final", "edge_id", "action", 
                   "date_started", "operation_time", "topological_map", "final_node",
                   "n_robots"]

    """
    
    
    #turn of warning
    pd.options.mode.chained_assignment = None  # default='warn'
    
    #1) initialise counter column
    df["n_robots"] = 0    
    tic = time.time()
    
    for i in range(len(df)):
        if i%5000 == 0:
            toc = time.time()
            print(i, "iterations in", toc-tic, "secs")
        
        #2) find edges that count towards congestion
        #All edges towards target + target_origin
        edge_list = []
        target = df["target"][i]
        origin = df["origin"][i]
        edge_list.append(target + "_" + origin)
        for wp in adjacent[target]:
            edge_list.append(wp + "_" + target)
        
        #3) Only need to consider data for same run_id
        subset = df[df["run_id"]==df["run_id"][i]]
        
        #4) Count total no. of robots on edges that count for congestion
        #includes the robot that is taking data
        subset = subset.loc[subset["edge_id"].isin(edge_list),:] #use isin
        criterion = (subset['date_started'] >= df["date_started"][i]) & (subset['date_started'] <= df["date_finished"][i])
        df["n_robots"][i] = subset.loc[criterion]["agent"].nunique()
        
    toc = time.time()
    print("Time taken(get_congestion):", toc-tic, "secs")
    
    return df