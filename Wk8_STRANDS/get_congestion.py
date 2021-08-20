# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:40:39 2021

@author: pyliu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:34:04 2021

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
    Add a column to the dataframe to show how many robots (in total) are on an edge at any time

    Parameters
    ----------
    df : Pandas DataFrame
        Initial DataFrame 

    Returns
    -------
    df : Pandas DataFrame
        Altered DataFrame 

    """
    pd.options.mode.chained_assignment = None  # default='warn'

    tic = time.time()
    df["n_robots"] = 0
    
    for i in range(len(df)):
        if i%5000 == 0:
            toc = time.time()
            print(i, "iterations in", toc-tic, "seconds")
        (df['date_started'] >= df["date_started"][i]) & (df['date_started'] <= df["date_finished"][i])
        
        edge_list = []
        edge_list.append(df["edge_id"][i])
        target = underscore_suffix(df["edge_id"][i])
        for origin in adjacent[target]:
            edge = origin + "_" + target
            edge_list.append(edge)
        criterion = (df['date_started'] >= df["date_started"][i]) & (df['date_started'] <= df["date_finished"][i]) & (df["edge_id"].isin(edge_list))
        
        df["n_robots"][i] = len(df.loc[criterion])
    
    toc = time.time()
    print("Time taken (get_congestion):", toc-tic, "seconds")
    return df