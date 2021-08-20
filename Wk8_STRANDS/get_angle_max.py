# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 14:16:36 2021

@author: pyliu
"""

import yaml
import pandas as pd
import numpy as np
import scipy as sp
import time

from underscore_prefix import *
from underscore_suffix import *
from calc_angle import *
from calc_length import *

from get_context import *
from get_adjacent import *
from get_coords import *

def get_angle_max(filename = "aaf_map.yaml", suppress_message = False):
    """
    Finds max angle exiting edges in a topological map

    Parameters
    ----------
    filename : STR
        name of topological map
    suppress_message : BOOL
        If True, do not print the time taken for this program to run

    Returns
    -------
    angle_max : DICT
        Key is edge_id
        value is max angle in radians
    """

    tic = time.time()
    
    #turn off warnings
    pd.options.mode.chained_assignment = None  # default='warn'
    
    context = get_context(filename, suppress_message = True)
    adjacent = get_adjacent(filename, suppress_message = True)
    coords = get_coords(filename, suppress_message = True)

    angle_max = {}
    for i in range(len(context)):
        origin1 = underscore_prefix(context["edge_id"][i])
        target1 = underscore_suffix(context["edge_id"][i])
        max1 = 0
        for node in adjacent[target1]:
            if node != origin1:
                angle = np.pi - calc_angle(coords[origin1], coords[target1], coords[node])
                if angle > max1:
                    max1 = angle
        angle_max[context["edge_id"][i]] = max1
        
    toc = time.time()
    if suppress_message == False:
        print("Time taken (get_angle_max):", toc-tic, "secs")
    
    return angle_max