# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:52:13 2021

@author: pyliu
"""
import pandas as pd
import numpy as np
import math
import scipy as sp
import time
import yaml

from underscore_prefix import *
from underscore_suffix import *

def get_data_yaml(filename = "blenheim_random_success.yaml"):
    """
    Extract data from YAML file to pandas DataFrame

    Parameters
    ----------
    filename : STR
        name of yaml file containing data
        The default is "blenheim_random_success.yaml".

    Returns
    -------
    df : pandas df
        columns=["origin", "target", "edge_id", "operation_time", "n_robots"].

    """
    tic = time.time()
    
    #1) Get data from yaml
    with open(filename) as file:
        documents = yaml.full_load(file)
    
    #2) create empty df
    total_length = 0
    for edge in documents:
        for n in range(len( documents[edge] )):
            total_length += len( documents[edge][n] )
    df = pd.DataFrame(index = np.arange(total_length), columns=["origin", "target", "edge_id", "operation_time", "n_robots"])
    
    #3) add data to df
    ind = 0
    for edge in documents:
        origin = underscore_prefix(edge)
        target = underscore_suffix(edge)
        for n in range(len( documents[edge] )):
            for i in range(len( documents[edge][n] )):
                t = documents[edge][n][i]
                df["origin"][ind] = origin
                df["target"][ind] = target
                df["edge_id"][ind] = edge
                df["operation_time"][ind] = t
                df["n_robots"][ind] = n+1
                ind += 1
    toc = time.time()
    print("Time taken:", toc-tic, "secs")
    return df
    