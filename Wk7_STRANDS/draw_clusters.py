# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 12:24:26 2021

@author: pyliu
"""

import networkx as nx
import yaml
import pandas as pd
import numpy as np
import scipy as sp

def draw_clusters(filename, clusters):
    """
    Draw network visualisation graph from a .YAML topological map

    Parameters
    ----------
    filename : STR
        Filename of a .YAML file. 
    clusters : pandas DataFrame
        Cluster labels for edges. columns = ["edge_id", "cluster_id"]
        
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
    all_colors = ["r", "g", "b", "c", "m", "yellow", 
                  "orange", "purple", "lime", "cyan", "darkgray", 
                  "pink", "brown", "olive", "gold", "teal", "maroon"]
    if np.max(clusters["cluster_id"]) + 1 > len(all_colors):
        raise ValueError("Too many clusters. Max no. of clusters: ", len(all_colors))
        
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
        if nodes[i][:8] == "WayPoint":
            G.add_node(nodes[i][8:], pos=(node_coords[i][0], node_coords[i][1]), color = "orange" )
        elif nodes[i] == "Frisoer" or nodes[i] == "Feuerloescher":
            G.add_node(nodes[i][:3], pos=(node_coords[i][0], node_coords[i][1]), color = "lightgreen" )
        else:
            G.add_node(nodes[i][0]+ nodes[i][-1], pos=(node_coords[i][0], node_coords[i][1]), color = "lightgreen" )

    
    #add edges
    count = 0
    for i in range(len(nodes)):
        for j in range(len(adjacent_nodes[i])):

            if nodes[i][:8] == "WayPoint":
                origin = nodes[i][8:]
            elif nodes[i] == "Frisoer" or nodes[i] == "Feuerloescher":
                origin = nodes[i][:3]
            else:
                origin = nodes[i][0]+ nodes[i][-1]
                
            if adjacent_nodes[i][j][:8] == "WayPoint":
                target = adjacent_nodes[i][j][8:]
            elif adjacent_nodes[i][j] == "Frisoer" or adjacent_nodes[i][j]  == "Feuerloescher":
                target = adjacent_nodes[i][j][:3]
            else:
                target = adjacent_nodes[i][j][0] + adjacent_nodes[i][j][-1]
            
            edge = nodes[i] + "_" + adjacent_nodes[i][j]
            while edge[0]== " ":
                edge = edge[1:]
            while edge[-1]== " ":
                edge = edge[:-1]   
            
            #pick colour according to cluster
            if edge in list(clusters['edge_id']):  
                col_ind = clusters.loc[clusters['edge_id'] == edge]
                col_ind = int(col_ind["cluster_id"])
                edge_col = all_colors[col_ind]
                G.add_edge(origin, target, color = edge_col, weight = 1)

            else:
                edge_col = "k"
                G.add_edge(origin, target, color = edge_col, weight = 0.2)
            
            count+= 1
    print("no. of edges:", count)

    #draw
    colors = nx.get_edge_attributes(G,'color').values()
    
    weights = nx.get_edge_attributes(G,'weight').values()
    nx.draw(G, nx.get_node_attributes(G, 'pos'), 
            edge_color = colors,
            width=list(weights),
            with_labels=True, node_color = "orange", node_size = 50, font_size = 5,
            arrows = True, connectionstyle='arc3, rad = 0.05')
    
    for i in range( np.max(clusters["cluster_id"]) + 1):
        print("CLuster", i, "has colour code:", all_colors[i])
    print("Edges without enough data are black")
    return G