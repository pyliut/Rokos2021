# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:33:32 2021

@author: pyliu
"""
import pandas as pd
import numpy as np
import math
import scipy as sp
import time

from select_data_edge import *
from underscore_prefix import *
from underscore_suffix import *
from get_context import *

def distances_length(context):
    """
    Returns difference matrix between edges and a list of the edge names
    Difference metric is difference between edge lengths

    Parameters
    ----------
    context : Pandas DataFrame
        Columns are "edge_id", "origin", "target", "edge_length", "n_connections_origin","n_connections_target"

    Returns
    -------
    diff_matrix : 2D array
        Difference metric between edges
    edge_list : array
        List of edge_id

    """
    #turn off warnings
    pd.options.mode.chained_assignment = None  # default='warn'
    
    tic = time.time()
    
    edge_list = []
    diff_matrix = []
    for i in range(len(context)):
        edge_list.append(context["edge_id"][i])
        diff_list = []
        for j in range(len(context)):
            diff = np.abs( context["edge_length"][i] - context["edge_length"][j] )
            diff_list.append(diff)
    
        diff_matrix.append(diff_list)
        
    toc = time.time()
    print("Time taken: ", toc-tic, "secs")
    
    return diff_matrix, edge_list