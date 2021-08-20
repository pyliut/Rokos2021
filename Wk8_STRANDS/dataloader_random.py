# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 18:27:31 2021

@author: pyliu
"""
import pandas as pd
import numpy as np
import scipy as sp
import time

def dataloader_random(df_class_diff, balanced = True, suppress_message = False):
    """
    Randomise entries 
    Balance numbers of each class if balanced == True

    Parameters
    ----------
    df_class_diff : pandas df
        columns = ["edge1", "edge2", "same_cluster",
                   "edge_length_diff", "origin_connections_diff", 
                  "target_connections_diff", "total_connections_diff", 
                  "max_angle_diff", "sum_angle_diff"]
    balanced : BOOL
        If True, we draw the same number of class 0 as class 1. The default is True.
    suppress_message : BOOL
        If True, do not print message

    Returns
    -------
    df_class_random : pandas df
        columns = ["edge1", "edge2", "same_cluster",
                   "edge_length_diff", "origin_connections_diff", 
                  "target_connections_diff", "total_connections_diff", 
                  "max_angle_diff", "sum_angle_diff"]

    """
    
    
    #1) Split by binary class
    df_0 = df_class_diff[df_class_diff["same_cluster"]==0]
    df_1 = df_class_diff[df_class_diff["same_cluster"]==1]
    
    #2) Randomise entries
    if balanced == True:
        if len(df_0) < len(df_1):
            df_class_random = pd.concat([df_0, df_1[:len(df_0)]])
        else:
            df_class_random = pd.concat([df_1, df_0[:len(df_1)]])
        df_class_random = df_class_random.sample(frac = 1).reset_index(drop = True)
    
    else:
        df_class_random = pd.concat([df_1, df_0])
        df_class_random = df_class_random.sample(frac = 1).reset_index(drop = True)
    
    if suppress_message == False:
        print("Samples drawn:", len(df_class_random))
    
    return df_class_random

