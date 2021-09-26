# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 13:39:28 2021

@author: pyliu
"""
import pandas as pd

def get_valid2(df, remove_multimodal = True):
    
    #1) remove data where target == final_node
    df = df[df["target"] == df["final_node"]]
    
    #2) remove unsuccessful runs
    df = df[df["status"] == "success"]
    
    #3) remove multimodal edges
    if remove_multimodal == True:
        df = df[df["n_robots"] == 1]
    
    #4) remove unneccessary columns
    df = df[["origin", "target", "edge_id","time_to_waypoint","operation_time"]]
    
    return df
    