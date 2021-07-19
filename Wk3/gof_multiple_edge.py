# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:52:03 2021

@author: pyliu
"""
import pandas as pd
import scipy as sp
from scipy import stats
import numpy as np


from update_mean import *
from goodness_of_fit2 import *
from select_data_edge import *
from underscore_prefix import *
from underscore_suffix import *


def gof_multiple_edge(df, model_list = ["norm", "lognorm"], n_edge = 1,  n_iter = 1):
    
    
    pd.options.mode.chained_assignment = None  # default='warn'
    
    #1) initialise empty pandas DataFrame
    n_dist = len(model_list)
    errors = pd.DataFrame(index = np.arange(n_dist),columns = ["Model","KS statistic", "KS p-value", "MAE", "MSE"])
    errors["Model"] = None
    errors["KS statistic"] = 0.0
    errors["KS p-value"] = 0.0
    errors["MAE"] = 0.0
    errors["MSE"] = 0.0
    
    #2) select 1 edge from the input df
    #2a) Order the data in terms of edges with the largest no. of observations
    count = df["edge_id"].value_counts()        #sort by no. of samples per edge
    count.to_csv('waypoint_pairs.csv')          #save and reload for DataFrame format
    count = pd.read_csv("waypoint_pairs.csv")
    count.columns = ["edge_id", "samples"]      #rename columns
    
    #2b) split edge_id into target & origin
    count["origin"] = None          #Add new columns
    count["target"] = None
    
    for i in range(len(count["edge_id"])):
        count["origin"][i] = underscore_prefix(str(count["edge_id"][i]))
        count["target"][i] = underscore_suffix(str(count["edge_id"][i]))
    
    for j in range(n_edge):
        #2c) Select data of interest
        wp1 = count["origin"][j]
        wp2 = count["target"][j]
        #2d) Select edge in one direction
        edge = str(wp1) + "_" + str(wp2)
        print(j, edge)
        subset = select_data_edge(df, edge)
        #2e) independent variable to plot over
        t_op = subset["operation_time"]
        
        #3) iterate over all models in scipy
        for i, d in enumerate(model_list):
            model = getattr(sp.stats, d)
            D_mean, p_mean, mae_mean, mse_mean = goodness_of_fit2(t_op, n_iter = n_iter, model = model)
            
            errors["Model"][i] = d
            errors["KS statistic"][i] += D_mean
            errors["KS p-value"][i] += p_mean
            errors["MAE"][i] += mae_mean
            errors["MSE"][i] += mse_mean

    errors["KS statistic"] /= n_edge
    errors["KS p-value"] /= n_edge
    errors["MAE"] /= n_edge
    errors["MSE"] /= n_edge
            
    return errors
        
        
        
        
        