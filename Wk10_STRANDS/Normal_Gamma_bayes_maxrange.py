# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 12:47:58 2021

@author: pyliu
"""
import scipy as sp
import numpy as np
import pandas as pd

from select_data_edge import *


def Normal_Gamma_bayes_maxrange(df, metric = "difference"):
    """
    Find maximum mean & variance for Bayesian lognormal fitting

    Parameters
    ----------
    df : pandas df
        observations
    metric : metric : STR
        Valid metrics are ["operation_time","difference"]. The default is "difference".

    Raises
    ------
    ValueError
        Invalid metric

    Returns
    -------
    max_range : TYPE
        Input for Normal_Gamma_bayes_nothreshold

    """
    edges = df["edge_id"].unique()
    max_range = 0
    for edge in edges:
        subset = select_data_edge(df, edge)
        if metric == "operation_time":
            t_op = subset["operation_time"]
        elif metric == "difference":
            t_op = subset["operation_time"] - subset["time_to_waypoint"]
        else:
            raise ValueError("Invalid metric. Valid metrics are ['operation_time','difference']")

        offset = np.min(t_op)-0.01
        t_obs = np.log(t_op - offset)
        max_current = np.max(t_obs)
        if max_current > max_range:
            max_range = max_current
    return max_range