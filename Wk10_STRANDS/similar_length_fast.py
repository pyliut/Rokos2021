# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 17:28:57 2021

@author: pyliu
"""

import pandas as pd
import numpy as np
import scipy as sp
import time

from get_length import *
from select_data_edge import *


def similar_length_fast(edge, df_train, length_train, length_test, cutoff = 1):
    """
    Get n_similar most similar edges from training map by edge length

    Parameters
    ----------
    edge : STR
        test edge name
    filename_test : STR
        Name of file containing topological map data for test map. The default is "aaf_map.yaml".
    filename_train : STR
        Name of file containing topological map data for train map. The default is "tsc_map.yaml".
    n_similar : INT
        number of similar edges returned. The default is 1.

    Returns
    -------
    STR
        most similar edge by edge length

    """

    #2) initialise vars
    edges_train = list(length_train.keys())
    length_edge = length_test[edge]
    
    #3) calculate edge length diff
    length_diff = [np.abs(length_edge - length_train[train]) for train in edges_train]
    
    index = np.array(length_diff).argmin()
    edge_train = edges_train[index]
    subset_train = select_data_edge(df_train,edge_train)
    while len(subset_train) < cutoff:
        length_diff.pop(index)
        edges_train.pop(index)
        if len(length_diff) < 1:
            raise ValueError("Error: no data")
        index = np.array(length_diff).argmin()
        edge_train = edges_train[index]
        subset_train = select_data_edge(df_train,edge_train)

    return edge_train
    