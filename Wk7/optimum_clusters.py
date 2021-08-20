# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 14:13:12 2021

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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

def optimum_clusters(distance_matrix, max_clusters = 15, method = "ss"):
    """
    Returns score for each choice of n_cluster

    Parameters
    ----------
    distance_matrix : TYPE
        DESCRIPTION.
    max_clusters : INT, scalar
        Maximum number of clusters to test. The default is 15.
    method : STR
        Choose from ["ss", "ch", "db"], where "ss" uses silhouette score, "ch" uses Calinski-Harabasz index, "db" uses Davies-Bouldin index. 
        The default is "ss".

    Raises
    ------
    ValueError
        When method is invalid.
        Can only be one of ["ss", "ch", "db"]

    Returns
    -------
    n_clusters : INT, vector
        np.arange(2, max_cluster + 1)
    s_list : FLOAT, vector
        score for each number of clusters

    """
    
    n_clusters = np.arange(2,max_clusters,1)
    s_list = []             #score list
    for i in n_clusters:
        clustering = AgglomerativeClustering(n_clusters=i, affinity='precomputed', linkage = 'average')
        labels = clustering.fit_predict( distance_matrix )
        if method == "ss":
            s = silhouette_score(distance_matrix , labels, metric='euclidean')
        elif method == "ch":
            s = calinski_harabasz_score(distance_matrix , labels)
        elif method == "db":
            s = davies_bouldin_score(distance_matrix , labels)
        else:
            raise ValueError("Method can be one of ['ss','ch','db']")
        s_list.append(s)
    
    plt.plot(n_clusters,s_list)
    plt.xlabel("n_clusters")
    plt.xlim([n_clusters[0], n_clusters[-1]]);
    if method == "ss":
        plt.title("Silhouette Coefficient")
        plt.ylabel("Silhouette coeff")
        print("Optimum no. of clusters:",n_clusters[np.argmax(s_list)])
        print("Max silhouette coeff:", np.max(s_list))
    elif method == "ch":
        plt.title("Calinski-Harabasz Index")
        plt.ylabel("CH Index")
        print("Optimum no. of clusters:",n_clusters[np.argmax(s_list)])
        print("Min CH index:", np.max(s_list))
    elif method == "db":
        plt.title("Davies-Bouldin Index")
        plt.ylabel("DB Index")
        print("Optimum no. of clusters:",n_clusters[np.argmin(s_list)])
        print("Max DB index:", np.min(s_list))
    else:
        raise ValueError("Method can be one of ['ss','ch','db']")     
    
    return n_clusters, s_list