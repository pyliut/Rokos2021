# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:21:18 2021

@author: pyliu
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp


#helper functions
def find_sigmoid(z):
  return 1/(1+np.exp(-z))

def find_loss(y,y_hat):
  return np.sum(-y*np.log(y_hat)-(1-y)*np.log(1-y_hat),axis=1)

def find_h(X,w,w0):
  return w@X + w0

def logistic_regression(X, y, n_iter = 1000, lr = 0.0001, plot_graph = True):
    
    
    #initialise weight vector
    w = np.random.randn(1,X.shape[0])
    w0 = 0

    losses = []
    
    for i in range(n_iter):
        h = find_h(X,w,w0)
        h = h.astype(float)
        y_hat = find_sigmoid(h)
    
        #grad is (y_hat-y)*(x_j)
    
        grad_w = (y_hat-y) @ X.T
        grad_w0 = np.sum((y_hat-y), axis=1, keepdims=True)
    
    
    
        w = w-lr*grad_w
        w0= w0-lr*grad_w0
    
        #keep loss for plotting vs iter
        losses.append(find_loss(y,y_hat))
    
    if plot_graph == True:
        print("final weight vector: w0 then w")
        print(w0,w)
        plt.plot(losses)
    
    return w0, w
    