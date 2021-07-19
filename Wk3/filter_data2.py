# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:34:04 2021

@author: pyliu
"""

import pandas as pd
from datetime import datetime
from datetime import timedelta

def filter_data2(df):
    pd.options.mode.chained_assignment = None  # default='warn'

    df["n_robots"] = 0
    
    for i, robots in enumerate(df["n_robots"]):
        wp1 = df["origin"][i]
        wp2 = df["target"][i]
        subset = df.loc[df["origin"].isin([wp1, wp2]) & df["target"].isin([wp1, wp2]),["edge_id", "start"]]
        for j, t_start in enumerate(subset["start"]):
            if t_start >= df["start"][i] and t_start <= df["finish"][i]:
                df["n_robots"][i] += 1
    
    return df