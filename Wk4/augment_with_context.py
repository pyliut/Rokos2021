# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 12:31:52 2021

@author: pyliu
"""
import pandas as pd
import numpy as np
import scipy as sp

def augment_with_context(similar_edges, context):
    """
    Adds information from context to similar_edges

    Parameters
    ----------
    similar_edges : Pandas DataFrame
        columns  = "edge_1", "edge_2","count_1","count_2", "ks_statistic", "p_value"
    context : Pandas DataFrame
        Columns = "edge_id", "origin", "target", "edge_length", "n_connections_origin","n_connections_target"

    Returns
    -------
    similar_edges : Pandas DataFrame
        Columns = "edge_1", "edge_2","count_1","count_2", "ks_statistic", "p_value", 
                    "edge_length_1", "edge_length_2", "origin_connections_1", "origin_connections_2",
                    "target_connections_1", "target_connections_2",
                    "length_diff", "origin_connections_diff", "target_connections_diff"
        
    """
    #turn off warnings
    pd.options.mode.chained_assignment = None  # default='warn'
    
    #1) initialise new cols
    similar_edges["edge_length_1"] = None
    similar_edges["edge_length_2"] = None
    similar_edges["origin_connections_1"] = None
    similar_edges["origin_connections_2"] = None
    similar_edges["target_connections_1"] = None
    similar_edges["target_connections_2"] = None
    
    similar_edges["length_diff"] = None
    similar_edges["origin_connections_diff"] = None
    similar_edges["target_connections_diff"] = None
    
    #2) Add from context
    for i in range(len(similar_edges)):
        edge_1 = similar_edges["edge_1"][i]
        edge_2 = similar_edges["edge_2"][i]
        
        similar_edges["edge_length_1"][i] = float ( context.loc[context['edge_id'] == edge_1]["edge_length"] )
        similar_edges["edge_length_2"][i] = float ( context.loc[context['edge_id'] == edge_2]["edge_length"] )
        
        similar_edges["origin_connections_1"][i] = float ( context.loc[context['edge_id'] == edge_1]["n_connections_origin"] )
        similar_edges["origin_connections_2"][i] = float ( context.loc[context['edge_id'] == edge_2]["n_connections_origin"] )
        
        similar_edges["target_connections_1"][i] = float ( context.loc[context['edge_id'] == edge_1]["n_connections_target"] )
        similar_edges["target_connections_2"][i] = float ( context.loc[context['edge_id'] == edge_2]["n_connections_target"] )
    
        similar_edges["length_diff"][i] = np.abs(similar_edges["edge_length_1"][i] - similar_edges["edge_length_2"][i])
        similar_edges["origin_connections_diff"][i] = np.abs(similar_edges["origin_connections_1"][i] - similar_edges["origin_connections_2"][i])
        similar_edges["target_connections_diff"][i] = np.abs(similar_edges["target_connections_1"][i] - similar_edges["target_connections_2"][i])
        
    return similar_edges