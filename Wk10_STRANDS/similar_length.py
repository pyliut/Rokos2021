# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:58:04 2021

@author: pyliu
"""
import pandas as pd
import numpy as np
import scipy as sp
import time

from get_length import *


def similar_length(edge, filename_test = "aaf_map.yaml", filename_train = "tsc_map.yaml", n_similar=1):
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
    STR, vector
        n_similar most similar edges by edge length

    """
    #1) get contexts
    length_test = get_length(filename_test, suppress_message=True)
    length_train = get_length(filename_train, suppress_message=True)
    
    #2) initialise vars
    edges_train = list(length_train.keys())
    length_diff = []
    length_edge = length_test[edge]
    
    #3) calculate edge length diff
    for train in edges_train:
        length_diff.append(np.abs(length_edge - length_train[train]))
    #length_diff = [np.abs(length_edge - length_train[train]) for train in edges_train]
    
    #4) get n_similar edges with smallest edge length
    similar_indices = list(np.argpartition(length_diff,n_similar)[:n_similar])
    
    edges_train=np.array(edges_train)
    length_diff = np.array(length_diff)
    return list(edges_train[similar_indices]), list(length_diff[similar_indices])
    