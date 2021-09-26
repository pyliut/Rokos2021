# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:39:38 2021

@author: pyliu
"""
import pandas as pd
import numpy as np

def get_valid(df, remove_multimodal = True, remove_initial = True):
    """
    Use after get_congestion
    Final filtering step: get_data, get_adjacent, get_congestion, get_valid

    Parameters
    ----------
    df : pandas df
        columns = ["_id", "origin", "target", "succeeded", "run_id", "date_finished",
                   "policy_goal", "agent", "_meta", "is_final", "edge_id", "action", 
                   "date_started", "operation_time", "topological_map", "final_node",
                   "n_robots"]

    remove_multimodal : BOOL
        If True, only returns data with n_robots == 1. The default is True.
    
    remove_initial : BOOL
        If True, only returns data where is_initial is False. The default is True.

    Returns
    -------
    df : pandas df
        columns = ["origin", "target", "edge_id", "operation_time", "start", "finish", "n_robots"]

    """
    
    if remove_initial == True:
        #1) add a column to state whether the origin node is the first node in a journey
        df = df.sort_values(["run_id","agent","date_finished"], ascending = (True, True, True))
        
        #restate the index in the sorted order
        ind = np.arange(len(df["is_final"]))
        df = df.set_index(ind)
        
        df["is_initial"] = False
        pd.options.mode.chained_assignment = None  # default='warn'
        for i in range(len(df["is_final"])-1):
            if df["is_final"][i] == True:
                df["is_initial"][i+1] = True
        
        #2a) remove data where is_initial == TRUE (Do this at the END)
        df = df[ df["is_initial"] == False ]
    
    #2b) remove data where is_final == TRUE (Do this at the END)
    df = df[ df["is_final"] == False ]
    
    #3) remove unsuccessful runs (Do this at the END)
    df = df[ df["succeeded"] == True ]
    
    #4) remove multimodal edges
    if remove_multimodal == True:
        df = df[ df["n_robots"] == 1 ]
    
    #4) remove unnecessary columns
    df = df.loc[:,["origin", "target", "edge_id", "operation_time", "n_robots"]]
    
    
    return df