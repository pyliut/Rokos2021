# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:54:13 2021

@author: pyliu
"""


def select_data_edge(df, edge = "WayPoint69_WayPoint70"):
    """

    Parameters
    ----------
    df : Pandas DataFrame
        Contains output of filter_data(df).
    edge : STR
        Specifies edge_id. The default is "WayPoint69_WayPoint70".
    
    Returns
    -------
    subset : Pandas DataFrame
        DESCRIPTION.

    """
    subset = df.loc[df["edge_id"].isin([edge]), :]
    return subset