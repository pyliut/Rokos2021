# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 22:55:39 2021

@author: pyliu
"""
import pandas as pd
import numpy as np
import scipy as sp
import sklearn
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

from dataloader_random import *

def classifier_featureimportance(df_train, balanced = True):
    """
    Get importance of features

    Parameters
    ----------
    df_train : pandas df
        training data
    balanced : pandas df
        If true, we use balanced data (equal class 0 & class 1). The default is True.

    Returns
    -------
    importance : OBJECT
        feature importance (between 0 & 1) 
        features are ["edge_length_diff", "origin_connections_diff", 
                                  "target_connections_diff", "total_connections_diff", 
                                  "max_angle_diff", "sum_angle_diff"]

    """

    df_train = dataloader_random(df_train, balanced = balanced)
    
    #class labels
    y = np.array(df_train["same_cluster"])
    y = y.astype(float)
    #attributes
    X = np.array(df_train[["edge_length_diff", "origin_connections_diff", 
                                  "target_connections_diff", "total_connections_diff", 
                                  "max_angle_diff", "sum_angle_diff"]])
        
    
    #1) Create Classifier
    #criterion{“gini”, “entropy”}, default=”gini”. The function to measure the quality of a split.
    #max_depthint, default=None. The maximum depth of the tree. 
    
    clf=RandomForestClassifier(n_estimators=100)
    
    #2) Train 
    clf.fit(X,y)
    
    #3) Show importance
    feature_names = ["edge_length_diff", "origin_connections_diff", 
                                  "target_connections_diff", "total_connections_diff", 
                                  "max_angle_diff", "sum_angle_diff"]
    importance = pd.Series(clf.feature_importances_,index=feature_names).sort_values(ascending=False)
    print(importance)
    
    y_pos = [i for i in range(len(importance.index))]
    plt.barh(y_pos, importance);
    plt.xlabel("Feature importance score")
    plt.ylabel("Features")
    plt.title("Compare feature importance")
    plt.yticks(y_pos, importance.index);
    plt.savefig("recent_FeatureImportance.png", bbox_inches="tight", dpi = 1000)
    
    return importance