# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 17:17:06 2021

@author: pyliu
"""
import scipy as sp
import pandas as pd
import numpy as np
import time

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor


def regressor_nsimilar(clf,df_features_seen,df_features_unseen,n_similar = 5,
                       features = ["edge_length_diff", "origin_connections_diff", 
                                  "target_connections_diff", "total_connections_diff", 
                                  "max_angle_diff", "sum_angle_diff"]):
    """
    Finds the most similar edges in training map to edges in test set 
    (according to predicted KS)

    Parameters
    ----------
    clf : object
        Pre-trained classifier
    df_features_seen : pandas df
        Training map
        columns = ["edge_id","edge_length","sum_angle","max_angle",
                 "origin_connections","target_connections","total_connections"])
    df_features_unseen : pandas df
        Test map
        columns = ["edge_id","edge_length","sum_angle","max_angle",
                 "origin_connections","target_connections","total_connections"])
    n_similar : INT
        Number of similar edges to find. The default is 5.
    features : STR, array
        DESCRIPTION. The default is ["edge_length_diff", "origin_connections_diff",  
                                     "target_connections_diff", "total_connections_diff",  
                                     "max_angle_diff", "sum_angle_diff"].

    Returns
    -------
    df_similar : pandas df
        columns = ["edge1","edge2","similar_edge","similar_ks", "all_edge", "all_ks"])
    

    """
    
    
    tic = time.time()
    #1) get edges
    edges_seen = list(df_features_seen["edge_id"])
    edges_unseen = list(df_features_unseen["edge_id"])
    
    #2) create empty df to store result
    df_similar = pd.DataFrame(index = np.arange(len(edges_unseen)),
                              columns = ["edge_id","similar_edge","similar_ks", "all_edge", "all_ks"])
    
    #3) Use classifier to predict KS score
    for i, edge1 in enumerate(edges_unseen):
        edge_list = []
        ks_list = []
        X = []
        if i % 25 == 0:
            toc = time.time()
            print(i,"edges:",toc-tic,"secs")
            
        for edge2 in edges_seen:
            X_current = []
            entry1 = df_features_unseen[df_features_unseen["edge_id"] == edge1]
            entry2 = df_features_seen[df_features_seen["edge_id"] == edge2]
            
            for feature in features:
                feature = feature[:-5]
                X_current.append( np.abs(float(entry1[feature]) - float(entry2[feature]) ) )
            
            edge_list.append(edge2)
            X.append(X_current)
        ks_list = np.array(clf.predict(X))
        edge_list = np.array(edge_list)
                
        
        similar_indices = list(np.argpartition(ks_list,n_similar)[:n_similar])
        df_similar["edge_id"][i] = edge1
        df_similar["all_ks"][i] = ks_list
        df_similar["all_edge"][i] = edge_list
        df_similar["similar_ks"][i] = ks_list[similar_indices]
        df_similar["similar_edge"][i] = edge_list[similar_indices]
    
    toc = time.time()
    print("Time taken (regressor_nsimilar):",toc-tic,"secs")
    
    return df_similar
    