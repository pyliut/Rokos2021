# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:47:16 2021

@author: pyliu
"""
import math
import numpy as np

from Gaussian_broadcast import *
from Gamma import *


def Normal_Gamma(mean, var, mu_0, beta, a, b):
    """
    Calculate normal-gamma distribution

    Parameters
    ----------
    mean : FLOAT/pandas Series, scalar or vector
        mean 
    var : FLOAT/pandas Series, scalar or vector
        variance
    mu_0 : FLOAT, scalar
        parameter of Normal-Gamma distribution
    beta : FLOAT, scalar
        parameter of Normal-Gamma distribution
    a : FLOAT, scalar
        parameter of Normal-Gamma distribution
    b : FLOAT, scalar
        parameter of Normal-Gamma distribution

    Returns
    -------
    FLOAT/pandas Series
        Calculation of Normal-Gamma distribution probability


    """
    
    p1 = Gaussian_broadcast(mean, mu_0, np.divide(var,beta))
    p2 = Gamma(1/var,a,b)
    return np.multiply(p1,p2)