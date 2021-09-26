# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:36:50 2021

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
from get_adjacent import *
from get_coords import *

from calc_angle import *
from calc_length import *

def distances_angle_max(filename = "aaf_map.yaml"):
    """
    Returns difference matrix between edges and a list of the edge names
    Difference metric is difference between maximum angle of turning

    Parameters
    ----------
    filename : STR
        name of topological map
        
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
    
    context = get_context(filename)
    adjacent = get_adjacent(filename)
    coords = get_coords(filename)
    
    edge_list = []
    diff_matrix = []
    
    for i in range(len(context)):
        edge_list.append(context["edge_id"][i])
        origin1 = underscore_prefix(context["edge_id"][i])
        target1 = underscore_suffix(context["edge_id"][i])
        max1 = 0
        for node in adjacent[target1]:
            if node != origin1:
                angle = np.pi - calc_angle(coords[origin1], coords[target1], coords[node])
                if angle > max1:
                    max1 = angle
        
        diff_list = []
        for j in range(len(context)):
            origin2 = underscore_prefix(context["edge_id"][j])
            target2 = underscore_suffix(context["edge_id"][j])
            max2 = 0
            for node in adjacent[target2]:
                if node != origin2:
                    angle = np.pi - calc_angle(coords[origin2], coords[target2], coords[node])
                    if angle > max2:
                        max2 = angle
            
            diff = np.abs( max1 - max2 )
            diff_list.append(diff)
    
        diff_matrix.append(diff_list)
        
    toc = time.time()
    print("Time taken (distances_angle_max): ", toc-tic, "secs")
    
    return diff_matrix, edge_list