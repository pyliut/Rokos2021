# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 13:50:44 2021

@author: pyliu
"""
import yaml
import pandas as pd
import numpy as np
import scipy as sp
import time

from get_context import *


def get_length(filename = "aaf_map.yaml", suppress_message = False):
    """
    Finds length of edges in a topological map

    Parameters
    ----------
    filename : STR
        name of topological map
    suppress_message : BOOL
        If True, do not print the time taken for this program to run

    Returns
    -------
    length : DICT
        Key is edge_id
        value is edge_length
    """

    tic = time.time()
    
    context = get_context(filename, suppress_message = True)
    
    #turn off warnings
    pd.options.mode.chained_assignment = None  # default='warn'
    length = {}
    for i in range(len(context)):
        length[context["edge_id"][i]] = context["edge_length"][i]
    
    toc = time.time()
    if suppress_message == False:
        print("Time taken (get_length):", toc-tic, "secs")
    
    return length