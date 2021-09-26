# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 16:11:09 2021

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

def classifier_pointtest(df_train,df_test,
                         features = ["edge_length_diff", "origin_connections_diff", 
                                  "target_connections_diff", "total_connections_diff", 
                                  "max_angle_diff", "sum_angle_diff"], 
                        target = "same_cluster",
                        classifier = "rf",balanced_train = True, balanced_test = False):
    """
    Print accuracy & classification report for a specified classification method
    Valid classifiers are ["rf", "dt", "knn", "lr", "svm", "gb", "ab",  "nb", "qda"]
    Returns trained classifier
    
    Parameters
    ----------
    df_train : pandas df
        Trains binary classifier
    df_test : pandas df
        Tests binary classifier
    classifier : STR
        Determines type of classifier used
        Valid classifiers are ["rf", "dt", "knn", "lr", "svm", "gb", "ab", "nb", "qda"]. 
        "rf" is random forest. "dt" is decision tree. "knn" is K nearest neighbours (K=1). 
        "lr" is logistic regression."svm" is support vector machine. "gb" is gradient boosting.
        "ab" is ada boost."nb" is naive bayes", "qda" is quadratic discriminant analysis
        The default is "random forest".
    balanced : BOOL
        If True, use a balanced draw from both the train & test datasets. The default is True.

    Returns
    -------
    clf : object
        trained classifier

    """

    #1) Get dataset into correct format
    if target == "same_cluster":
        df_train = dataloader_random(df_train, balanced = balanced_train)
        df_test = dataloader_random(df_test, balanced = balanced_test)
    
    #2) Train classifier
    if classifier == "dt":
        clf = DecisionTreeClassifier(criterion = "gini", max_depth = None)
    elif classifier == "lr":
        clf = LogisticRegression(random_state=0)
    elif classifier == "svm":
        clf = svm.SVC(kernel='rbf') 
    elif classifier == "rf":
        clf = RandomForestClassifier(n_estimators=500, max_features = 0.55, max_depth = 14)
    elif classifier == "knn":
        clf = KNeighborsClassifier(n_neighbors=1)
    elif classifier == "gb":
        clf = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 1, random_state = 0)
    elif classifier == "ab":
        clf = AdaBoostClassifier(n_estimators = 100, learning_rate = 1, random_state = 0)
    elif classifier == "nb":
        clf = GaussianNB()
    elif classifier == "qda":
        clf = QuadraticDiscriminantAnalysis()
    
    else:
        raise ValueError('Invalid classifier. Valid classifiers are ["rf", "dt", "knn", "lr", "svm", "gb", "ab", "nb", "qda"]')

    
    #class labels
    y = np.array(df_train[target]).astype(float)
    
    y_test = np.array(df_test[target]).astype(float)
    
    #attributes
    X = np.array(df_train[features])
    
    X_test = np.array(df_test[features])
        
    
    clf = clf.fit(X,y)
    
    
    #5) Predict the response for test dataset
    y_pred = clf.predict(X_test)
    
    accuracy = sklearn.metrics.accuracy_score(y_test,y_pred)
    print("Classifier type:", classifier)
    print("Accuracy:", accuracy)
    print(sklearn.metrics.classification_report(y_test,y_pred))
    sklearn.metrics.plot_roc_curve(clf, X_test, y_test)
    plt.show()
    
    return clf