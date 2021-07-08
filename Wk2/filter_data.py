# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:27:41 2021

@author: pyliu
"""
import pandas as pd

def filter_data(df):
    """

    Parameters
    ----------
    df : Pandas DataFrame
        Initial DataFrame from MongoDB

    Returns
    -------
    df : Pandas DataFrame
        Preliminary filtering

    """
    #1) we previously removed null & unsuccessful entries
    
    #2) remove data where policy_goal = target
    df = df[ df["policy_goal"] != df["target"] ]
    
    #3) remove every column except: origin, target, edge_id, operation_time
    df = df.loc[:,["origin", "target", "edge_id", "operation_time"]]
    return df
