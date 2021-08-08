# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 14:51:00 2021

@author: pyliu
"""

import numpy as np

def train_regression(nb, x_train, y_train):
    """
    Fits a polynomial model to data

    Parameters
    ----------
    nb : INT, scalar
        order of polynomial used for fitting
    x_train : FLOAT, vector
        independent variable for training
    y_train : FLOAT, vector
        dependent variable for training

    Returns
    -------
    w : FLOAT, vector
        contains coefficients of the fitted polynomial

    """

    #-1 means we do not care what the value is
    x = np.reshape(x_train,(-1,1))
      
    #w_0 + w1*x + w2*x^2 + ... w_nb*x^nb
    basis = np.arange(nb+1)
      
    #we want basis to be a row vector
    basis = np.reshape(basis,(1,-1))
      
    # A = [x^0, x^1, ... x^nb]
    A_train = x**basis
      
      
    # @ is matrix multiplication
    # w = (A^T A) ^(-1) * (A^T b)
    w = np.linalg.inv(A_train.T@A_train) @ (A_train.T @ y_train)
      
    return w