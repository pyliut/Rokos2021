# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:52:03 2021

@author: pyliu
"""
import pandas as pd
import scipy as sp
from scipy import stats
import numpy as np
import time


from update_mean import *
from gof import *
from select_data_edge import *
from underscore_prefix import *
from underscore_suffix import *


def gof_multiple_edge(df, model_list = ["norm", "lognorm"], n_edge = 1):
    
    tic = time.time()
    
    pd.options.mode.chained_assignment = None  # default='warn'
    
    #1) initialise empty pandas DataFrame
    n_dist = len(model_list)
    errors = pd.DataFrame(index = np.arange(n_dist),columns = ["Model","KS statistic", "KS p-value"])
    errors["Model"] = None
    errors["KS statistic"] = 0.0
    errors["KS p-value"] = 0.0
    
    #2) select 1 edge from the input df
    #2a) Order the data in terms of edges with the largest no. of observations
    count = df["edge_id"].value_counts()        #sort by no. of samples per edge
    count = count.to_frame("samples").reset_index()
    count.columns = ["edge_id", "samples"]      #rename columns
    
    for j in range(n_edge):
        #2c) Select data of interest
        edge = count["edge_id"][j]
        subset = select_data_edge(df, edge)
        #2e) independent variable to plot over
        t_op = subset["operation_time"] - subset["time_to_waypoint"] 
        
        #3) iterate over all models in scipy
        for i, d in enumerate(model_list):
            model = getattr(sp.stats, d)
            D, p= gof(t_op, model = model)
            
            errors["Model"][i] = d
            errors["KS statistic"][i] += D
            errors["KS p-value"][i] += p
            
        toc = time.time()
        print(j, edge, toc-tic, "secs")

    errors["KS statistic"] /= n_edge
    errors["KS p-value"] /= n_edge
    
    toc = time.time()
    print("Time taken:", toc-tic, "secs")
            
    return errors
        
        
        
        
        