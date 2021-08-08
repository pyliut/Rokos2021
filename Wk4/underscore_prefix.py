# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 12:28:31 2021

@author: pyliu
"""

def underscore_prefix(edge_id):
    """
    Input an edge_id, e.g. "WayPoint69_WayPoint70"
    Returns the characters before "_"

    Parameters
    ----------
    edge_id : STR
        2 WayPoints separated by a "_"

    Returns
    -------
    STR
        Substring of edge_id before "_"

    """
    ind = edge_id.find("_")
    return edge_id[:ind]