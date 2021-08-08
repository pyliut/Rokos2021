# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:02:19 2021

@author: pyliu
"""
import pandas as pd
import numpy as np
import math
import scipy as sp
import time

from select_data_edge import *
from underscore_prefix import *
from underscore_suffix import *

def ks_between_edges(df, threshold = 0.05, min_samples = 20, similar = True):
    

    tic = time.time()
    #turn off warnings
    pd.options.mode.chained_assignment = None  # default='warn'
    
    #1) Order edges by amount of data
    #1a) Order the data in terms of edges with the largest no. of observations
    count = df["edge_id"].value_counts()        #sort by no. of samples per edge
    count.to_csv('waypoint_pairs.csv')          #save and reload for DataFrame format
    count = pd.read_csv("waypoint_pairs.csv")
    count.columns = ["edge_id", "samples"]      #rename columns
    
    #1b) split edge_id into target & origin
    count["origin"] = None          #Add new columns
    count["target"] = None
    
    for i in range(len(count["edge_id"])):
        count["origin"][i] = underscore_prefix(str(count["edge_id"][i]))
        count["target"][i] = underscore_suffix(str(count["edge_id"][i]))
    
    #2a) initialise empty pandas df
    n_edge = len(count)
    edge_series = count["edge_id"]
    edge_list = edge_series.tolist()
    edge_list.insert(0,"edge_id")
    
    ks_matrix = pd.DataFrame(index = np.arange(n_edge),columns = edge_list)
    ks_matrix["edge_id"] = edge_list[1:]
    
    p_matrix = pd.DataFrame(index = np.arange(n_edge),columns = edge_list)
    p_matrix["edge_id"] = edge_list[1:]
    
    combined_matrix = pd.DataFrame(index = np.arange(n_edge),columns = edge_list)
    combined_matrix["edge_id"] = edge_list[1:]
    
    #2b) create a list to store similar edges
    similar_edges = []
    
    #3) calculate ks tuple
    for i in range(n_edge):
        for j in range(i+1):
            #3a) Select first edge
            wp1 = count["origin"][i]
            wp2 = count["target"][i]
            edge1 = str(wp1) + "_" + str(wp2)
            subset1 = select_data_edge(df, edge1)
            t_op1 = subset1["operation_time"]
            
            #3b) Select second edge
            wp1 = count["origin"][j]
            wp2 = count["target"][j]
            edge2 = str(wp1) + "_" + str(wp2)
            subset2 = select_data_edge(df, edge2)
            t_op2 = subset2["operation_time"]
            
            #calculate ks statistic & p-value between edges
            ks_stat, p_val = sp.stats.ks_2samp(t_op1,t_op2)
            
            ks_matrix[edge_list[i+1]][j] = ks_stat
            p_matrix[edge_list[i+1]][j] = p_val
            combined_matrix[edge_list[i+1]][j] = [ks_stat, p_val]
            if similar == True:  
                if p_val > threshold and p_val < 1 and count["samples"][i] > min_samples and count["samples"][j] > min_samples:
                    similar_edges.append([edge1, edge2,count["samples"][i],count["samples"][j], ks_stat, p_val])
            elif similar == False:
                if p_val < threshold and p_val >= 0 and count["samples"][i] > min_samples and count["samples"][j] > min_samples:
                    similar_edges.append([edge1, edge2,count["samples"][i],count["samples"][j], ks_stat, p_val])
            
                
    #4) Save to excel
    #4a) Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter("ks_between_edges.xlsx", engine='xlsxwriter')
    
    #4b) Write each dataframe to a different worksheet.
    ks_matrix.to_excel(writer, sheet_name='ks_stat')
    p_matrix.to_excel(writer, sheet_name='p_val')
    combined_matrix.to_excel(writer, sheet_name='combined')
    
    #4c) Close the Pandas Excel writer and output the Excel file.
    writer.save()
    
    #5) return a dataframe containing the edges that are similar
    similar = pd.DataFrame(similar_edges, columns = ["edge_1", "edge_2","count_1","count_2", "ks_statistic", "p_value"])
    similar_sorted = similar.sort_values("ks_statistic", ascending = True)
    similar_sorted.head()     
    
    toc = time.time()
    print("Time taken:", toc-tic, "secs")
    return similar_sorted