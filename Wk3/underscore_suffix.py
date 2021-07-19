# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 12:35:02 2021

@author: pyliu
"""

def underscore_suffix(edge_id):
    """
    Input an edge_id, e.g. "WayPoint69_WayPoint70"
    Returns the characters after "_"

    Parameters
    ----------
    edge_id : STR
        2 WayPoints separated by a "_"

    Returns
    -------
    STR
        Substring of edge_id after "_"

    """
    ind = edge_id.find("_")
    return edge_id[ind+1:]