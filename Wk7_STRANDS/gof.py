# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 14:08:44 2021

@author: pyliu
"""

import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


from update_mean import *

def gof(t_op, model = stats.norm):
    """
    Calculates KS statistic and p value for a given model

    Parameters
    ----------
    t_op : FLOAT, vector
        observed durations
    model : Scipy model
        Scipy-parameterised model. The default is stats.norm.

    Raises
    ------
    ValueError
        If the scipy model has too many parameters (i.e. more than 5)

    Returns
    -------
    D : FLOAT, scalar
        KS statistic
    p : FLOAT, scalar
        KS p-value

    """
    #1) Fit model
    params = model.fit(t_op)
    
    #2a) K-S Test

    if len(params) == 1:
        D,p = sp.stats.kstest(t_op, lambda k: model.cdf(k, params[0]))
    elif len(params) == 2:
        D,p = sp.stats.kstest(t_op, lambda k: model.cdf(k, params[0], params[1]))
    elif len(params) == 3:
        D,p = sp.stats.kstest(t_op, lambda k: model.cdf(k, params[0], params[1], params[2]))
    elif len(params) == 4:
        D,p = sp.stats.kstest(t_op, lambda k: model.cdf(k, params[0], params[1], params[2], params[3]))
    elif len(params) == 5:
        D,p = sp.stats.kstest(t_op, lambda k: model.cdf(k, params[0], params[1], params[2], params[3], params[4]))
    else:
        raise ValueError("too many params. Max = 5")
    
    
    return D,p