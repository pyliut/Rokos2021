# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 13:24:13 2021

@author: pyliu
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import math
import scipy as sp
from scipy import stats

import sklearn
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

def optimum_clusters_DBSCAN(distance_matrix, method = "ss", min_samples = 1):
    """
    Returns score for each choice of n_cluster

    Parameters
    ----------
    distance_matrix : TYPE
        DESCRIPTION.
    method : STR
        Choose from ["ss", "ch", "db"], where "ss" uses silhouette score, "ch" uses Calinski-Harabasz index, "db" uses Davies-Bouldin index. 
        The default is "ss".
    min_samples : INT
        another tuning param for DBSCAN. The Default is 1.

    Raises
    ------
    ValueError
        When method is invalid.
        Can only be one of ["ss", "ch", "db"]

    Returns
    -------
    tune_list : INT, vector
        tuning param
    s_list : FLOAT, vector
        score for each value of tuning parameter

    """
    samples = np.arange(1,min_samples+1,1)
    tune = np.arange(0.03,0.99,0.01)
    s_dict = {}
    tune_dict = {}
    
    for j in samples:
        s_list = []             #score list
        tune_list = []          #corresponding tuning params
        for i in tune:
            clustering = DBSCAN(eps = i, min_samples = j, metric='precomputed')
        
            labels = clustering.fit_predict( distance_matrix )
            if len(np.unique(labels)) < 2:
                continue
            if method == "ss":
                s = silhouette_score(distance_matrix , labels, metric='euclidean')
            elif method == "ch":
                s = calinski_harabasz_score(distance_matrix , labels)
            elif method == "db":
                s = davies_bouldin_score(distance_matrix , labels)
            else:
                raise ValueError("Method can be one of ['ss','ch','db']")
            s_list.append(s)
            tune_list.append(i)
        s_dict[str(j)] = s_list
        tune_dict[str(j)] = tune_list
    
        plt.plot(tune_list,s_list)
        if method == "ss":
            print("min_samples:",j)
            print("Optimum tuning param:",np.round(tune_list[np.argmax(s_list)],4))
            print("Max SS:", np.round(np.max(s_list),4))
        elif method == "ch":
            print("min_samples:",j)
            print("Optimum tuning param:",np.round(tune_list[np.argmax(s_list)],4))
            print("Max CH:", np.round(np.max(s_list),4))
        elif method == "db":
            print("min_samples:",j)
            print("Optimum tuning param:",np.round(tune_list[np.argmin(s_list)],4))
            print("Min DB:", np.round(np.min(s_list),4))
        else:
            raise ValueError("Method can be one of ['ss','ch','db']")     
    plt.xlabel("tuning param")
    plt.xlim([tune_list[0], tune_list[-1]]);
    plt.legend(samples)
    if method == "ss":
        plt.title("Silhouette Coefficient")
        plt.ylabel("Silhouette coeff")
    elif method == "ch":
        plt.title("Calinski-Harabasz Index")
        plt.ylabel("CH Index")
    elif method == "db":
        plt.title("Davies-Bouldin Index")
        plt.ylabel("DB Index")
    else:
        raise ValueError("Method can be one of ['ss','ch','db']")     
    
    return tune_dict, s_dict