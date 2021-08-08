# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 16:27:04 2021

@author: pyliu
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import math
import scipy as sp

from select_data_edge import *
from Lognormal import *
from Normal_Gamma_bayes import *


def mergefit(df, edge_list, n_samples = 10):
    """
    Performs mergefit:
    Fits a lognormal distribution based on n_samples datapoints from each edge in edge_list

    Parameters
    ----------
    df : pandas dataframe
        columns = ["origin", "target", "edge_id", "operation_time"]
    edge_list : STR, vector
        list of "edge_id" in a particular cluster
    n_samples : INT, scalar
        number of datapoints taken from each edge in mergefit. The default is 10.

    Returns
    -------
    mean_map : FLOAT, scalar
        parameter of lognormal distribution
    var_map : FLOAT, scalar
        parameter of lognormal distribution
    offset : FLOAT, scalar
        parameter of lognormal distribution

    """
    #1) take n_samples datapoints from each edge
    t_op = []
    indices = []
    for edge in edge_list:
        subset = select_data_edge(df, edge)
        #independent variable to plot over
        t = list(subset["operation_time"])
        max_ind = min(n_samples, len(t)//2)
        t = t[0:max_ind]
        indices.append(max_ind)
        t_op = [*t_op, *t]
    
    #2) fit model
    #use offset and take log of data
    offset= np.min(t_op) - 0.01
    t_log = np.log(t_op - offset)
    
    #n_terms = 1 if you want just the initial estimates
    n_terms = len(t_log)    
    t_obs = t_log[0:n_terms]
    
    #set parameters
    mu_0 = 1
    beta = 0.1
    a = 1
    b = 1
    
    #Bayesian MAP estimate of mean & variance of Gaussian distribution
    mean_test, var_test, posterior, mean_map, var_map = Normal_Gamma_bayes(t_obs, mu_0, beta, a, b, plot_graph = False)
    return mean_map, var_map, offset