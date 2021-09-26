# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 12:56:00 2021

@author: pyliu
"""

import yaml
import pandas as pd
import numpy as np
import scipy as sp
import time

from select_data_edge import *
from Gamma import *
from Lognormal import *
from Gaussian import *
from Gaussian_broadcast import *
from Normal_Gamma import *

from Normal_Gamma_bayes import *
from Normal_Gamma_bayes_update import *

def evaluate_groundtruth(df, filename = "aaf_map.yaml",metric = "difference", 
                         cutoff = 1, prior_params = [1,0.1,1,1], verbose = False):
    """
    Evaluate KS stat for MLE & Bayesian fitting using all data

    Parameters
    ----------
    df : pandas df
        observations
    filename : STR
        Name of file containing topological map data. The default is "aaf_map.yaml".
    metric : STR
        Valid metrics are ["operation_time","difference"]. The default is "difference".
    cutoff : INT
        Minimum number of datapoints to perform fitting. The default is 1.
    prior_params : FLOAT, vector
        4-element array. Elements are [mu_0, beta, a , b]. The default is [1,0.1,1,1].
    verbose : BOOL
        print progress information if True. The default is True.

    Raises
    ------
    ValueError
        Invalid metric

    Returns
    -------
    fit : pandas df
        columns = ["edge_id", "origin", "target","n_obs",
                   "ks_bayes","s_bayes","loc_bayes","scale_bayes",
                   "ks_mle","s_mle","loc_mle","scale_mle"]

    """
    
    tic = time.time()
    
    #1) initialise empty lists to store data
    nodes = []
    adjacent_nodes = []
    
    #2) open yaml file and extract information
    with open(filename) as file:
        documents = yaml.full_load(file)
        for i in range(len(documents)):
            #Waypoint of current node
            nodes.append(documents[i]["meta"]["node"])
            
            #Waypoints of connecting nodes
            adjacent_nodes_temp = []
            for j in range(len(documents[i]["node"]["edges"])):
                adjacent_nodes_temp.append(documents[i]["node"]["edges"][j]["node"])
            adjacent_nodes.append(adjacent_nodes_temp)
    
    #3) Initialise pandas df to store data
    #turn off warnings
    pd.options.mode.chained_assignment = None  # default='warn'
        
    #initialise pd df
    n_edge = 0
    for i in range(len(adjacent_nodes)):
        n_edge += len(adjacent_nodes[i])
    fit = pd.DataFrame(index = np.arange(n_edge), columns = ["edge_id", "origin", "target","n_obs",
                                                             "ks_bayes","s_bayes","loc_bayes","scale_bayes",
                                                             "ks_mle","s_mle","loc_mle","scale_mle"] )
    
    #4) Store edge_id, origin, target
    index = 0
    for i in range(len(nodes)):
        for j in range(len(adjacent_nodes[i])):
            fit["origin"][index] = nodes[i]
            fit["target"][index] = adjacent_nodes[i][j]
            
            fit["edge_id"][index] = fit["origin"][index] + "_" + fit["target"][index]
            index += 1
    
    #5) set parameters for Bayesian fit
    mu_0 = prior_params[0]
    beta = prior_params[1]
    a = prior_params[2]
    b = prior_params[3]
            
    #6) Get data
    for i in range(len(fit)):
        if i % 20 == 0 and verbose == True:
            toc = time.time()
            print(i, "edges:", toc-tic, "secs")
        
        edge = fit["edge_id"][i]
        subset = select_data_edge(df,edge)
        fit["n_obs"][i] = len(subset)
        if metric == "operation_time":
            t_op = subset["operation_time"]
        elif metric == "difference":
            t_op = subset["operation_time"] - subset["time_to_waypoint"]
        else:
            raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")

        if len(subset) >= cutoff:
            
            #7) Bayesian fit
            offset= np.min(t_op) - 0.01
            t_obs = np.log(t_op - offset)
            #Bayesian MAP estimate of mean & variance of Gaussian distribution
            mean_test, var_test, posterior, mean_map, var_map = Normal_Gamma_bayes(t_obs, mu_0, beta, a, b, plot_graph = False)
            fit["s_bayes"][i] = np.sqrt(var_map)
            fit["loc_bayes"][i] = offset
            fit["scale_bayes"][i] = np.exp(mean_map)

            #8) Calculate KS (Bayesian)
            ks_bayes,pval_bayes = sp.stats.kstest(t_op, 
                lambda k: sp.stats.lognorm.cdf( k, s = fit["s_bayes"][i], loc = fit["loc_bayes"][i], scale = fit["scale_bayes"][i]) )
            fit["ks_bayes"][i] = ks_bayes
            
            #9) MLE fit
            params = sp.stats.lognorm.fit(t_op)
            fit["s_mle"][i] = params[0]
            fit["loc_mle"][i] = params[1]
            fit["scale_mle"][i] = params[2]
            
            #10) Calculate KS (MLE)
            ks_mle,pval_mle = sp.stats.kstest(t_op, 
                lambda k: sp.stats.lognorm.cdf( k, s = params[0], loc = params[1], scale = params[2]) )
            fit["ks_mle"][i] = ks_mle
            
        else:
            fit["s_bayes"][i] = np.NaN
            fit["loc_bayes"][i] = np.NaN
            fit["scale_bayes"][i] = np.NaN
            fit["ks_bayes"][i] = np.NaN
            fit["s_mle"][i] = np.NaN
            fit["loc_mle"][i] = np.NaN
            fit["scale_mle"][i] = np.NaN
            fit["ks_mle"][i] = np.NaN
    
    fit = fit.sort_values("n_obs",ascending = False).reset_index(drop=True)
            
    return fit
    