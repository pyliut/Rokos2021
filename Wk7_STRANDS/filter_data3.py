# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:39:38 2021

@author: pyliu
"""
import pandas as pd
import numpy as np

def filter_data3(df, remove_multimodal = True):
    """
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    remove_multimodal : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    
    #1) add a column to state whether the origin node is the first node in a journey
    df = df.sort_values(["run_id","agent","finish"], ascending = (True, True, True))
    
    #restate the index in the sorted order
    ind = np.arange(len(df["is_final"]))
    df = df.set_index(ind)
    
    df["is_initial"] = False
    pd.options.mode.chained_assignment = None  # default='warn'
    for i in range(len(df["is_final"])-1):
        if df["is_final"][i] == True:
            df["is_initial"][i+1] = True
    
    #2) remove data where is_final == TRUE (Do this at the END)
    df = df[ df["is_final"] == False ]
    
    #3) remove data where is_initial == TRUE (Do this at the END)
    df = df[ df["is_initial"] == False ]
    
    #4) remove multimodal edges
    if remove_multimodal == True:
        df = df[ df["n_robots"] == 1 ]
    
    #4) remove unnecessary columns
    df = df.loc[:,["origin", "target", "edge_id", "operation_time", "start", "finish", "n_robots", "is_final", "is_initial"]]
    
    
    return df