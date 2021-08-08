# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 12:03:34 2021

@author: pyliu
"""
import pandas as pd
import numpy as sp

from select_data_edge import *

def mergefit_augment_stats(mergefit_stats,df):
    """
    Augments mergefit_stats with differences between crps & ks scores

    Parameters
    ----------
    mergefit_stats : pandas dataframe
        columns = ["edge_id", "crps", "ks", "crps_orig", "ks_orig"]
        crps & ks are statistics for mergefit wrt unseen data for each edge
        crps_orig & ks_orig are statistics for original distribution
    df : pandas dataframe
        columns = ["origin", "target", "edge_id", "operation_time"]

    Returns
    -------
    mergefit_df : pandas dataframe
        returns "rank_diff", "adjusted_diff", "abs_diff"
        columns = ["edge_id","n_samples", "crps","crps_rank", "ks", "ks_rank", "rank_diff", "crps_adjusted", "ks_adjusted", "adjusted_diff", "abs_diff"])
    
    """
    
    
    #1) Initialise empty dataframe to store augmented statistics
    mergefit_df = pd.DataFrame(columns = ["edge_id","n_samples", "crps","crps_rank", "ks", "ks_rank", "rank_diff", "crps_adjusted", "ks_adjusted", "adjusted_diff", "abs_diff"])
    
    #2) Add information from mergefit_stats
    mergefit_df["edge_id"] = mergefit_stats["edge_id"]
    mergefit_df["crps"] = mergefit_stats["crps"]
    mergefit_df["ks"] = mergefit_stats["ks"]
    #only consider KS value (not p-value)
    for i in range(len(mergefit_df)):
        mergefit_df["ks"][i] = mergefit_df["ks"][i][0]
    
    #3) rank edges according to CRPS & KS score
    crps_ordered = np.sort( np.array(mergefit_df["crps"]) )
    ks_ordered = np.sort( np.array(mergefit_df["ks"]) )
    crps_ordered = np.sort( np.array(mergefit_df["crps"]) )
    ks_ordered = np.sort( np.array(mergefit_df["ks"]) )
    
    #4) add augmented information
    for i in range(len(mergefit_df)):
        #4a) number of observations at each edge
        mergefit_df["n_samples"][i] = len(select_data_edge(df, mergefit_df["edge_id"][i]))
        #4b) rank edges according to CRPS & KS score
        # calculate difference between CRPS & KS ranks for each edge
        mergefit_df["crps_rank"][i] = np.where(crps_ordered == mergefit_df["crps"][i])[0][0]
        mergefit_df["ks_rank"][i] = np.where(ks_ordered == mergefit_df["ks"][i])[0][0]
        mergefit_df["rank_diff"][i] = np.abs(mergefit_df["crps_rank"][i] - mergefit_df["ks_rank"][i])
        #4c) scale CRPS & KS scores so that they are between 0 & 1
        #calculate difference between adjusted CRPS & KS stats
        mergefit_df["crps_adjusted"][i] = mergefit_df["crps"][i]/crps_ordered[-1]
        mergefit_df["ks_adjusted"][i] = mergefit_df["ks"][i]/ks_ordered[-1]
        mergefit_df["adjusted_diff"][i] = np.abs(mergefit_df["crps_adjusted"][i] - mergefit_df["ks_adjusted"][i])
        #4d) calculate difference between actual CRPS & KS stats
        mergefit_df["abs_diff"][i] = np.abs(mergefit_df["crps"][i] - mergefit_df["ks"][i])
        
    mergefit_df.to_excel("mergefit_stats.xlsx")
    return mergefit_df