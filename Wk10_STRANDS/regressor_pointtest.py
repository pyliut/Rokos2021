# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 15:30:01 2021

@author: pyliu
"""
import pandas as pd
import numpy as np
import scipy as sp
import sklearn

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor



def regressor_pointtest(df_train, df_test, 
                        features = ["edge_length_diff", "origin_connections_diff", 
                                  "target_connections_diff", "total_connections_diff", 
                                  "max_angle_diff", "sum_angle_diff"], 
                        target = "ks",
                        classifier= "rf"):
    """
    Prints MSE & MAE of regression
    Returns trained classifier
    
    Parameters
    ----------
    df_train : pandas df
        Trains binary classifier
    df_test : pandas df
        Tests binary classifier
    features : TYPE, optional
        DESCRIPTION. The default is ["edge_length_diff", "origin_connections_diff",                                  "target_connections_diff", "total_connections_diff",                                  "max_angle_diff", "sum_angle_diff"].
    classifier : STR
        Determines type of classifier used
        Valid classifiers are ['ols','svr','rf','ab','gb']. 
        "rf" is random forest. "ols" is decision tree. "ab" is ada boost.
        "lr" is logistic regression."svr" is support vector machine. "gb" is gradient boosting.
        The default is "random forest".
    
    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    mse : TYPE
        DESCRIPTION.
    mae : TYPE
        DESCRIPTION.

    """

    #1) Create classifier
    if classifier == "ols":
        clf = Ridge()
    elif classifier == "svr":
        clf = SVR(kernel = "rbf", C = 3, epsilon = 0.01)
    elif classifier == "rf":
        clf = RandomForestRegressor(n_estimators = 100)
    elif classifier == "ab":
        clf = AdaBoostRegressor(n_estimators = 100, learning_rate = 0.01)
    elif classifier== "gb":
        clf = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.01)
    else:
        raise ValueError("Invalid classifier. Valid classifiers are ['ols','svr','rf','ab','gb']")
    
    #2) Load train & test data
    y_train = np.array(df_train[target]).astype(float)
    y_test = np.array(df_test[target]).astype(float)
    X_train = np.array(df_train[features])
    X_test = np.array(df_test[features])
    
    #3) Fit & Predict
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    
    #4) Calculate error
    mse = np.sum(np.square(y_pred-y_test))/len(y_pred)
    mae = np.sum(np.abs(y_pred-y_test))/len(y_pred)
    
    print("MSE:",mse)
    print("MAE:",mae)
    print("Max/min y_pred:", np.max(y_pred), np.min(y_pred))
    print("Max/min y_test:", np.max(y_test), np.min(y_test))
        
    return clf