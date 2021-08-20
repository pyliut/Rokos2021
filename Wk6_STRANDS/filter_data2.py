# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:34:04 2021

@author: pyliu
"""

import pandas as pd
from datetime import datetime
from datetime import timedelta

def filter_data2(df, lookahead = 5):
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

    df["n_robots"] = 0
    
    for i in range(len(df)-lookahead):
        for j in range(1, lookahead + 1):
            if df["start"][j] >= df["start"][i] and df["start"][j] <= df["finish"][i]:
                df["n_robots"][i] += 1
    
    return df