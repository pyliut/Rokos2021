# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 17:34:49 2021

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

def distances_connections(context):
    """
    Returns difference matrix between edges and a list of the edge names
    Difference metric is sum of differences between origin nodes and target nodes

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

            origin_connections_1= float ( context["n_connections_origin"][i] )
            origin_connections_2 = float ( context["n_connections_origin"][j] )
            
            target_connections_1= float ( context["n_connections_target"][i] )
            target_connections_2 = float ( context["n_connections_target"][j] )
        
            origin_connections_diff= np.abs(origin_connections_1 - origin_connections_2)
            target_connections_diff = np.abs(target_connections_1 - target_connections_2)
    
            diff = origin_connections_diff + target_connections_diff
            diff_list.append(diff)
    
        diff_matrix.append(diff_list)
        
    toc = time.time()
    print("Time taken: ", toc-tic, "secs")
    
    return diff_matrix, edge_list