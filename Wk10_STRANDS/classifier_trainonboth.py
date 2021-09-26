# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 23:51:18 2021

@author: pyliu
"""

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier

from dataloader_random import *

def classifier_trainonboth(df_train,df_test,balanced = True, n_iter = 5, step_size = 1000):
    """
    Generate graph of accuracies for different binary classification methods

    Parameters
    ----------
    df_train : pandas df
        Trains binary classifier
    df_test : pandas df
        Tests binary classifier
        Serves as a comparison test
    balanced : pandas df
        If true, we use balanced data (equal class 0 & class 1). The default is True.
    n_iter : INT
        number of iterations per calculation. We take the mean of all calculations. The default is 5.
    step_size : INT
        Increment in no. of samples. The default is 1000.

    Returns
    -------
    None.

    """
    # Unbalanced dataset
    tic = time.time()
    
    #1) Get dataset into correct format
    df_train = dataloader_random(df_train, balanced = balanced)
    df_test = dataloader_random(df_test, balanced = balanced)
    
    #2) Train classifier
    clf_lr_tsc = LogisticRegression(random_state=0)
    clf_svm_tsc = svm.SVC(kernel='rbf') 
    clf_rf_tsc = RandomForestClassifier(n_estimators=100)
    clf_lr_aaf = LogisticRegression(random_state=0)
    clf_svm_aaf = svm.SVC(kernel='rbf') 
    clf_rf_aaf = RandomForestClassifier(n_estimators=100)
    
    #3) select data for train & test
    n_train_range = np.arange(step_size,len(df_test),step_size)

    accuracy_lr_tsc = []
    accuracy_svm_tsc = []
    accuracy_rf_tsc = []
    accuracy_lr_aaf = []
    accuracy_svm_aaf = []
    accuracy_rf_aaf = []
    
    
    for n_train in n_train_range:
        #class labels
        y_train1 = np.array(df_train["same_cluster"])
        y_train1 = y_train1[0:n_train]
        y_train1 = y_train1.astype(float)
        
        y_train2 = np.array(df_test["same_cluster"])
        y_train2 = y_train2[0:n_train]
        y_train2 = y_train2.astype(float)
        
        y_test = np.array(df_test["same_cluster"])
        y_test = y_test[n_train:]
        y_test = y_test.astype(float)
    
        #attributes
        X_train1 = np.array(df_train[["edge_length_diff", "origin_connections_diff", 
                                      "target_connections_diff", "total_connections_diff", 
                                      "max_angle_diff", "sum_angle_diff"]][0:n_train])
        X_train2 = np.array(df_test[["edge_length_diff", "origin_connections_diff", 
                                      "target_connections_diff", "total_connections_diff", 
                                      "max_angle_diff", "sum_angle_diff"]][0:n_train])
        
        X_test = np.array(df_test[["edge_length_diff", "origin_connections_diff", 
                                  "target_connections_diff", "total_connections_diff", 
                                  "max_angle_diff", "sum_angle_diff"]][n_train:])
    
        mean_lr_tsc = 0
        mean_svm_tsc = 0
        mean_rf_tsc = 0
        mean_lr_aaf = 0
        mean_svm_aaf = 0
        mean_rf_aaf = 0
        
        for i in range(n_iter):
            #4) Train Decision Tree Classifer
            clf_lr_tsc = clf_lr_tsc.fit(X_train2,y_train2)
            clf_svm_tsc = clf_svm_tsc.fit(X_train2,y_train2)
            clf_rf_tsc = clf_rf_tsc.fit(X_train2,y_train2)
            clf_lr_aaf = clf_lr_aaf.fit(X_train1,y_train1)
            clf_svm_aaf = clf_svm_aaf.fit(X_train1,y_train1)
            clf_rf_aaf = clf_rf_aaf.fit(X_train1,y_train1)
    
            #5) Predict the response for test dataset
            y_pred_lr_tsc = clf_lr_tsc.predict(X_test)
            y_pred_svm_tsc = clf_svm_tsc.predict(X_test)
            y_pred_rf_tsc = clf_rf_tsc.predict(X_test)
            y_pred_lr_aaf = clf_lr_aaf.predict(X_test)
            y_pred_svm_aaf = clf_svm_aaf.predict(X_test)
            y_pred_rf_aaf = clf_rf_aaf.predict(X_test)
    
            #6) Evaluate model
            mean_lr_tsc += sklearn.metrics.accuracy_score(y_test, y_pred_lr_tsc)
            mean_svm_tsc += sklearn.metrics.accuracy_score(y_test, y_pred_svm_tsc)
            mean_rf_tsc += sklearn.metrics.accuracy_score(y_test,y_pred_rf_tsc)
            mean_lr_aaf += sklearn.metrics.accuracy_score(y_test, y_pred_lr_aaf)
            mean_svm_aaf += sklearn.metrics.accuracy_score(y_test, y_pred_svm_aaf)
            mean_rf_aaf += sklearn.metrics.accuracy_score(y_test,y_pred_rf_aaf)
        
        #7) take average
        accuracy_lr_tsc.append(mean_lr_tsc/n_iter)
        accuracy_svm_tsc.append(mean_svm_tsc/n_iter)
        accuracy_rf_tsc.append(mean_rf_tsc/n_iter)
        accuracy_lr_aaf.append(mean_lr_aaf/n_iter)
        accuracy_svm_aaf.append(mean_svm_aaf/n_iter)
        accuracy_rf_aaf.append(mean_rf_aaf/n_iter)
    
        toc = time.time()
        print(n_train, "datapoints:", toc-tic, "secs")
    
    toc = time.time()
    print("Time taken:", toc-tic, "secs")
        
    #8) Plot
    plt.plot(n_train_range, accuracy_lr_tsc, color = "green", linestyle = "dashed")
    plt.plot(n_train_range, accuracy_svm_tsc, color = "red", linestyle = "dashed")
    plt.plot(n_train_range, accuracy_rf_tsc, color = "purple", linestyle = "dashed")
    plt.plot(n_train_range, accuracy_lr_aaf, color = "green")
    plt.plot(n_train_range, accuracy_svm_aaf, color = "red")
    plt.plot(n_train_range, accuracy_rf_aaf, color = "purple")
    plt.xlabel("Training samples")
    plt.ylabel("Accuracy")
    plt.legend(["LR - Train on Test Map", "SVM - Train on Test Map", "RF - Train on Test Map",
               "LR - Train on Train Map", "SVM - Train on Train Map", "RF - Train on Train Map"])
    plt.title("Train classifiers on both maps")
    
    print( max( max(accuracy_lr_tsc),max(accuracy_lr_aaf),
                max(accuracy_svm_tsc),max(accuracy_svm_aaf),
                max(accuracy_rf_tsc),max(accuracy_rf_aaf) ) )
    plt.savefig("recent_TrainOnBoth.png", bbox_inches="tight", dpi = 1000)
    
    #save as dictionary
    accuracy = {}
    accuracy["lr_train"] = accuracy_lr_tsc
    accuracy["svm_train"] = accuracy_svm_tsc
    accuracy["rf_train"] = accuracy_rf_tsc
    accuracy["lr_test"] = accuracy_lr_aaf
    accuracy["svm_test"] = accuracy_svm_aaf
    accuracy["rf_test"] = accuracy_rf_aaf
    
    return accuracy