# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 23:17:31 2021

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis




from dataloader_random import *

def classifier_comparemethods(df_train,df_test,balanced = True, n_iter = 5, step_size = 1000):
    """
    Generate graph of accuracies for different binary classification methods
    ["rf", "dt", "knn", "lr", "svm", "gb", "ab",  "nb", "qda"]

    Parameters
    ----------
    df_train : pandas df
        Trains binary classifier
    df_test : pandas df
        Tests binary classifier
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
    clf_dt = DecisionTreeClassifier(criterion = "gini", max_depth = None)
    clf_lr = LogisticRegression(random_state=0)
    clf_svm = svm.SVC(kernel='rbf') 
    clf_rf = RandomForestClassifier(n_estimators=100)
    clf_knn = KNeighborsClassifier(n_neighbors=1)
    clf_gb = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 1, random_state = 0)
    clf_ab = AdaBoostClassifier(n_estimators = 100, learning_rate = 1, random_state = 0)
    clf_nb = GaussianNB()
    clf_qda = QuadraticDiscriminantAnalysis()
    

    #3) select data for train & test
    n_train_range = np.arange(step_size,len(df_train),step_size)

    accuracy_dt = []
    accuracy_lr = []
    accuracy_svm = []
    accuracy_rf = []
    accuracy_knn = []
    accuracy_gb = []
    accuracy_ab = []
    accuracy_nb = []
    accuracy_qda = []
    
    y_test = np.array(df_test["same_cluster"])
    y_test = y_test.astype(float)
    
    X_test = np.array(df_test[["edge_length_diff", "origin_connections_diff", 
                                  "target_connections_diff", "total_connections_diff", 
                                  "max_angle_diff", "sum_angle_diff"]])
    
    
    for n_train in n_train_range:
        #class labels
        y = np.array(df_train["same_cluster"])
        y = y[0:n_train]
        y = y.astype(float)
    
        #attributes
        X = np.array(df_train[["edge_length_diff", "origin_connections_diff", 
                                      "target_connections_diff", "total_connections_diff", 
                                      "max_angle_diff", "sum_angle_diff"]][0:n_train])
    
        mean_dt = 0
        mean_lr = 0
        mean_svm = 0
        mean_rf = 0
        mean_knn = 0
        mean_gb = 0
        mean_ab = 0
        mean_nb = 0
        mean_qda = 0
        for i in range(n_iter):
            #4) Train Decision Tree Classifer
            clf_dt = clf_dt.fit(X,y)
            clf_lr = clf_lr.fit(X,y)
            clf_svm = clf_svm.fit(X,y)
            clf_rf = clf_rf.fit(X,y)
            clf_knn = clf_knn.fit(X,y)
            clf_gb = clf_gb.fit(X,y)
            clf_ab = clf_ab.fit(X,y)
            clf_nb = clf_nb.fit(X,y)
            clf_qda = clf_qda.fit(X,y)
    
            #5) Predict the response for test dataset
            y_pred_dt = clf_dt.predict(X_test)
            y_pred_lr = clf_lr.predict(X_test)
            y_pred_svm = clf_svm.predict(X_test)
            y_pred_rf = clf_rf.predict(X_test)
            y_pred_knn = clf_knn.predict(X_test)
            y_pred_gb = clf_gb.predict(X_test)
            y_pred_ab = clf_ab.predict(X_test)
            y_pred_nb = clf_nb.predict(X_test)
            y_pred_qda = clf_qda.predict(X_test)
    
            #6) Evaluate model
            mean_dt += sklearn.metrics.accuracy_score(y_test, y_pred_dt)
            mean_lr += sklearn.metrics.accuracy_score(y_test, y_pred_lr)
            mean_svm += sklearn.metrics.accuracy_score(y_test, y_pred_svm)
            mean_rf += sklearn.metrics.accuracy_score(y_test,y_pred_rf)
            mean_knn += sklearn.metrics.accuracy_score(y_test,y_pred_knn)
            mean_gb += sklearn.metrics.accuracy_score(y_test,y_pred_gb)
            mean_ab += sklearn.metrics.accuracy_score(y_test,y_pred_ab)
            mean_nb += sklearn.metrics.accuracy_score(y_test,y_pred_nb)
            mean_qda += sklearn.metrics.accuracy_score(y_test,y_pred_qda)
        
        #7) take average
        accuracy_dt.append(mean_dt/n_iter)
        accuracy_lr.append(mean_lr/n_iter)
        accuracy_svm.append(mean_svm/n_iter)
        accuracy_rf.append(mean_rf/n_iter)
        accuracy_knn.append(mean_knn/n_iter)
        accuracy_gb.append(mean_gb/n_iter)
        accuracy_ab.append(mean_ab/n_iter)
        accuracy_nb.append(mean_nb/n_iter)
        accuracy_qda.append(mean_qda/n_iter)
    
        toc = time.time()
        print(n_train, "datapoints:", toc-tic, "secs")
    
    toc = time.time()
    print("Time taken:", toc-tic, "secs")
        
    #8) Plot
    plt.plot(n_train_range, accuracy_dt)
    plt.plot(n_train_range, accuracy_lr)
    plt.plot(n_train_range, accuracy_svm)
    plt.plot(n_train_range, accuracy_rf)
    plt.plot(n_train_range, accuracy_knn)
    plt.plot(n_train_range, accuracy_gb)
    plt.plot(n_train_range, accuracy_ab)
    plt.plot(n_train_range, accuracy_nb)
    plt.plot(n_train_range, accuracy_qda)
    plt.xlabel("Training samples")
    plt.ylabel("Accuracy")
    plt.legend(["Decision Tree", "LogReg", "SVM - rbf", "Random Forest", "KNN, K=1",
                "Gradient Boost", "Ada Boost", "Naive Bayes", "QDA"])
    plt.title("Compare Classification Methods")
    
    print( max( max(accuracy_rf),max(accuracy_lr),
                max(accuracy_svm),max(accuracy_knn),
                max(accuracy_dt), max(accuracy_gb),
                max(accuracy_ab), max(accuracy_nb),
                max(accuracy_qda) ) )
    plt.savefig("recent_CompareClassification.png", bbox_inches="tight", dpi = 1000)
    
    #save as dictionary
    accuracy = {}
    accuracy["knn"] = accuracy_knn
    accuracy["svm"] = accuracy_svm
    accuracy["lr"] = accuracy_lr
    accuracy["rf"] = accuracy_rf
    accuracy["dt"] = accuracy_dt
    accuracy["gb"] = accuracy_gb
    accuracy["ab"] = accuracy_ab
    accuracy["nb"] = accuracy_nb
    accuracy["qda"] = accuracy_qda
    
    return accuracy