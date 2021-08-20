# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:51:42 2021

@author: pyliu
"""
import numpy as np
import math
import scipy as sp


def calc_length(x,y):
    """
    Calculate length between 2 points - N dimensions

    Parameters
    ----------
    x : FLOAT, N-dim vector
        coords of point 1
    y : FLOAT, N-dim vector
        coords of point 2

    Returns
    -------
    length : FLOAT, scalar
        length between points 1 & 2

    """
    length = np.sqrt( np.sum( np.square(y-x) ) ) 
    return length