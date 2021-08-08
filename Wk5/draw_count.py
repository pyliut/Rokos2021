# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 11:13:56 2021

@author: pyliu
"""


import networkx as nx
import yaml
import pandas as pd
import numpy as np
import scipy as sp

def draw_count(filename, count):
    """
    Draw network visualisation graph from a .YAML topological map

    Parameters
    ----------
    filename : STR
        Filename of a .YAML file. 
    count : pandas DF
        columns = ["edge_id", "count"]
    Returns
    -------
    G : NetworkX graph
        Network visualisation graph

    """
    #1) initialise empty lists to store data
    nodes = []
    adjacent_nodes = []
    adjacent_coords = []
    node_coords = []
    edge_length = []
    
    #create colours for cluster_labels
    all_colors = ["r", "orange","g", "b", "purple"]
      
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
    
    #4) Draw Graph
    #initialise empty graph
    G = nx.MultiDiGraph()
    
    
    #add nodes
    for i in range(len(nodes)):
        G.add_node(nodes[i][8:], pos=(node_coords[i][0], node_coords[i][1]))
    counter = 0
    #add edges
    for i in range(len(nodes)):
        for j in range(len(adjacent_nodes[i])):
            origin = nodes[i][8:]
            target = adjacent_nodes[i][j][8:]
            edge = nodes[i] + "_" + adjacent_nodes[i][j]
            while edge[0]== " ":
                edge = edge[1:]
            while edge[-1]== " ":
                edge = edge[:-1]   
            
            #pick colour according to cluster
            if edge in list(count['edge_id']):  
                n_samples = count.loc[count['edge_id'] == edge]
                n_samples = int(n_samples["samples"])
                if n_samples >= 0 and n_samples < 20:
                    col_ind = 0
                elif n_samples >= 20 and n_samples < 50:
                    col_ind = 1
                elif n_samples >= 50 and n_samples < 100:
                    col_ind = 2
                elif n_samples >= 100 and n_samples < 250:
                    col_ind = 3
                elif n_samples >= 250:
                    col_ind = 4

                edge_col = all_colors[col_ind]
                G.add_edge(origin, target, color = edge_col, weight = 1)

            else:
                edge_col = "k"
                G.add_edge(origin, target, color = edge_col, weight = 0.2)
            
            counter += 1
    print("no. of edges:", counter)

    #draw
    colors = nx.get_edge_attributes(G,'color').values()
    weights = nx.get_edge_attributes(G,'weight').values()
    nx.draw(G, nx.get_node_attributes(G, 'pos'), 
            edge_color = colors,
            width=list(weights),
            with_labels=True, node_color = "orange", node_size = 150,
            arrows = True, connectionstyle='arc3, rad = 0.1')
    
    print("Legend:")
    print("Red < 20 samples")
    print("Orange < 50 samples")
    print("Green < 100 samples")
    print("Blue < 250 samples")
    print("Purple > 250 samples")
    return G