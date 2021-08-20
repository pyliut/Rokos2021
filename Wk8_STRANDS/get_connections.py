# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:06:42 2021

@author: pyliu
"""

import yaml
import pandas as pd
import numpy as np
import scipy as sp
import time

from get_context import *

def get_connections(filename = "aaf_map.yaml", suppress_message = False):
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
    connections : DICT
        Key is edge_id
        value is [n_connections_origin, n_connections_target]
    """
    
    tic = time.time()
    
    #turn off warnings
    pd.options.mode.chained_assignment = None  # default='warn'
    
    context = get_context(filename, suppress_message = True)
    
    connections = {}
    for i in range(len(context)):
        connections[context["edge_id"][i]] = [context["n_connections_origin"][i], context["n_connections_target"][i]]
    
    toc = time.time()
    if suppress_message == False:
        print("Time taken (get_connections):", toc-tic, "secs")
    
    return connections