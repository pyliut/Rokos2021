# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:32:07 2021

@author: pyliu
"""

import pandas as pd
from datetime import datetime
from datetime import timedelta

def filter_data1(df):
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
    
    #2) remove data where policy_goal = target (Do this at the END)
    #df = df[ df["policy_goal"] != df["target"] ]
    
    #3) remove every column except: origin, target, edge_id, is_final, operation_time, _meta, agent, run_id
    df = df.loc[:,["origin", "target", "final_node", "_id", "edge_id", "operation_time", "_meta"]]
    
    #4) make a column for the actual start and finish datetime objects
    df["start"] = datetime(2021,2,20,9,0,0)
    df["finish"] = datetime(2021,2,20,9,0,0)
    
    pd.options.mode.chained_assignment = None  # default='warn'
    for i, time in enumerate(df["_meta"]):
        t_finish = time["inserted_at"]
        df["finish"][i] = t_finish
        t_op = timedelta(seconds = df["operation_time"][1])
        df["start"][i] = t_finish - t_op
    
    #5) remove every column except: origin, target, edge_id, is_final, operation_time, start, finish
    df = df.loc[:,["origin", "target", "edge_id", "final_node", "operation_time", "start", "finish", "_id"]]
    
    return df