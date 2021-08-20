# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:58:16 2021

@author: pyliu
"""

import yaml
import pandas as pd
import numpy as np
import scipy as sp
import time

def get_adjacent(filename = "walmart_map.yaml"):
    """
    Infers context about edges in a topological map

    Parameters
    ----------
    filename : STR
        Filename of a .YAML file. The default is "walmart_map.yaml".

    Returns
    -------
    adjacent : DICT
        Key is a node (STR)
        Value is list of adjacent nodes
    """
    #1) initialise empty lists to store data
    nodes = []
    adjacent_nodes = []
    adjacent_coords = []
    node_coords = []
    edge_length = []
    
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
            
            #xy coords of current node
            node_coords.append([documents[i]["node"]["pose"]["position"]["x"], documents[i]["node"]["pose"]["position"]["y"]])
    
    #3) synthesize additional information about edge length and number of adjacent nodes
    for i in range(len(nodes)):
        coords_temp = []
        length_temp = []
        for adjacent in adjacent_nodes[i]:
            x1 = node_coords[i][0]
            y1 = node_coords[i][1]
            
            #xy coords of connecting nodes
            adjacent_ind = nodes.index(adjacent)
            x2 = node_coords[adjacent_ind][0]
            y2 = node_coords[adjacent_ind][1]
            coords_temp.append([x2,y2])
            
            #distance to connecting nodes (from current node)
            length = np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
            length_temp.append(length)
    
        adjacent_coords.append(coords_temp)
        edge_length.append(length_temp)
    
    toc = time.time()
    print("Time taken:", toc-tic, "secs")
    
    #4) Create a dictionary to store adjacent nodes to the current nodes
    adjacent = {}
    for i in range(len(nodes)):
        adjacent[nodes[i]] = adjacent_nodes[i]
    
    return adjacent