# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 12:09:21 2021

@author: pyliu
"""
import pandas as pd
from datetime import datetime
from datetime import timedelta
import time
def get_times(df, date_format = "%A, %B %d %Y, at %H:%M:%S hours"):
    
    pd.options.mode.chained_assignment = None  # default='warn'
    
    tic = time.time()
    
    for i in range(len(df)):
        temp = time.strptime(df["date_started"][i], date_format)
        df["date_started"][i] = datetime(*temp[:6])
        
        temp = time.strptime(df["date_at_node"][i], date_format)
        df["date_at_node"][i] = datetime(*temp[:6])
        
        temp = time.strptime(df["date_finished"][i], date_format)
        df["date_finished"][i] = datetime(*temp[:6])
        
        df["_meta"][i] = df["_meta"][i]["inserted_at"]
        
    toc = time.time()
    print("Time taken (get_times):", toc-tic, "secs")
    
    return df