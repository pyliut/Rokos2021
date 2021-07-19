# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:45:46 2021

@author: pyliu
"""

def select_data(df, wp1 = "WayPoint69", wp2 = "WayPoint70"):
    """

    Parameters
    ----------
    df : Pandas DataFrame
        Contains output of filter_data(df).
    wp1 : STR
        Specifies the 1st WayPoint. The default is "WayPoint69".
    wp2 : STR
        Specifies the 2nd WayPoint. The default is "WayPoint70".

    Returns
    -------
    subset : Pandas DataFrame
        DESCRIPTION.

    """
    subset = df.loc[df["origin"].isin([wp1, wp2]) & df["target"].isin([wp1, wp2]), ["edge_id", "operation_time"]]
    if subset.size == 0:
        print("ERROR: empty subset")
    return subset