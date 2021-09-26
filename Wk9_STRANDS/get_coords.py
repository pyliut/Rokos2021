# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:23:25 2021

@author: pyliu
"""

import yaml
import pandas as pd
import numpy as np
import scipy as sp
import time

def get_coords(filename = "walmart_map.yaml", suppress_message = False):
    """
    Infers context about edges in a topological map

    Parameters
    ----------
    filename : STR
        Filename of a .YAML file. The default is "walmart_map.yaml".
    suppress_message : BOOL
        If True, do not print the time taken for this program to run

    Returns
    -------
    context : Pandas DataFrame
        Columns are "edge_id", "origin", "target", "edge_length", "n_connections_origin","n_connections_target"

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
            node_coords.append([documents[i]["node"]["pose"]["position"]["x"], documents[i]["node"]["pose"]["position"]["y"], documents[i]["node"]["pose"]["position"]["z"]])
    
    #3) synthesize additional information about edge length and number of adjacent nodes
    for i in range(len(nodes)):
        coords_temp = []
        length_temp = []
        for adjacent in adjacent_nodes[i]:
            x1 = node_coords[i][0]
            y1 = node_coords[i][1]
            z1 = node_coords[i][2]
            
            #xy coords of connecting nodes
            adjacent_ind = nodes.index(adjacent)
            x2 = node_coords[adjacent_ind][0]
            y2 = node_coords[adjacent_ind][1]
            z2 = node_coords[adjacent_ind][2]
            coords_temp.append([x2,y2,z2])
            
            #distance to connecting nodes (from current node)
            length = np.sqrt( (x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
            length_temp.append(length)
    
        adjacent_coords.append(coords_temp)
        edge_length.append(length_temp)
    
    #4) Store data about EDGES as a pd df
    #turn off warnings
    pd.options.mode.chained_assignment = None  # default='warn'
        
    #initialise pd df
    n_edge = 0
    for i in range(len(adjacent_nodes)):
        n_edge += len(adjacent_nodes[i])
    context = pd.DataFrame(index = np.arange(n_edge), columns = ["edge_id", "origin", "target", "edge_length", "n_connections_origin","n_connections_target"] )
    
    index = 0
    for i in range(len(nodes)):
        for j in range(len(adjacent_nodes[i])):
            context["origin"][index] = nodes[i]
            context["target"][index] = adjacent_nodes[i][j]
            
            context["edge_id"][index] = context["origin"][index] + "_" + context["target"][index]
            context["edge_length"][index] = edge_length[i][j]
            
            context["n_connections_origin"][index] = len(adjacent_nodes[i])
            adjacent_ind = nodes.index(context["target"][index])
            context["n_connections_target"][index] = len(adjacent_nodes[adjacent_ind])
            index += 1
            
    #4) Create a dictionary to store coords to the current nodes
    coords = {}
    for i in range(len(nodes)):
        coords[nodes[i]] = node_coords[i]
        
        
    toc = time.time()
    if suppress_message == False:
        print("Time taken (get_coords):", toc-tic, "secs")
    
    return coords
        