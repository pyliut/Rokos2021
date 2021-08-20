# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 13:24:32 2021

@author: pyliu
"""
import pandas as pd

def get_valid1(df, adjacent):
    
    df = df.loc[ ( df["origin"].isin(adjacent.keys()) & df["target"].isin(adjacent.keys()) ), :]
    df = df[["_id", "status", "origin", "target", "edge_id","date_finished", "date_at_node", "date_started","_meta", "time_to_waypoint","operation_time", "final_node"]]
    df = df.reset_index(drop=True)
    return df