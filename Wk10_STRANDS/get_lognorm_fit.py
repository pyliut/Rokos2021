# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:44:12 2021

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

def get_lognorm_fit(df, filename = "aaf_map.yaml", metric = "difference"):
    """
    Get Bayesian optimisation params for all edges in a map

    Parameters
    ----------
    df : pandas df
        observation
    filename : STR
        name of YAML file containing map information. The default is "aaf_map.yaml".
    metric : STR
        Valid metrics are "difference" & "operation_time". The default is "difference".

    Raises
    ------
    ValueError
        invalid metric

    Returns
    -------
    fit : pandas df
        contains fitted params for Bayesian lognormal optimisation

    """
    #1) initialise empty lists to store data
    nodes = []
    adjacent_nodes = []

    
    tic = time.time()
    
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
            

    
    #4) Store data about EDGES as a pd df
    #turn off warnings
    pd.options.mode.chained_assignment = None  # default='warn'
        
    #initialise pd df
    n_edge = 0
    for i in range(len(adjacent_nodes)):
        n_edge += len(adjacent_nodes[i])
    fit = pd.DataFrame(index = np.arange(n_edge), columns = ["edge_id", "origin", "target","n_obs", "s","loc","scale"] )
    
    index = 0
    for i in range(len(nodes)):
        for j in range(len(adjacent_nodes[i])):
            fit["origin"][index] = nodes[i]
            fit["target"][index] = adjacent_nodes[i][j]
            
            fit["edge_id"][index] = fit["origin"][index] + "_" + fit["target"][index]
            index += 1
    for i in range(len(fit)):
        if i % 20 == 0:
            toc = time.time()
            print(i, "edges:", toc-tic, "secs")
        edge = fit["edge_id"][i]
        subset = select_data_edge(df, edge)
        #independent variable to plot over
        if metric == "difference":
            t_op = subset["operation_time"] - subset["time_to_waypoint"]
        elif metric == "operation_time":
            t_op = subset["operation_time"]
        else:
            raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")


        offset= np.min(t_op) - 0.01
        t_obs = np.log(t_op - offset)

        #set parameters
        mu_0 = 1
        beta = 0.1
        a = 1
        b = 1
        
        if len(t_obs) > 0:
            #Bayesian MAP estimate of mean & variance of Gaussian distribution
            mean_test, var_test, posterior, mean_map, var_map = Normal_Gamma_bayes(t_obs, mu_0, beta, a, b, plot_graph = False)

            fit["s"][i] = np.sqrt(var_map)
            fit["loc"][i] = offset
            fit["scale"][i] = np.exp(mean_map)
        else:
            fit["s"][i] = np.NaN
            fit["loc"][i] = np.NaN
            fit["scale"][i] = np.NaN
        fit["n_obs"][i] = len(t_op)
    
    toc = time.time()
    print("Time taken (get_lognorm_fit):", toc - tic, "secs")
    
    return fit
