# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 16:39:38 2021

@author: pyliu
"""

def filter_data3(df):
    
    #1) remove data where is_final = TRUE (Do this at the END)
    df = df[ df["is_final"] == False ]
    
    #2) remove multimodal edges
    df = df[ df["n_robots"] == 1 ]
    
    return df