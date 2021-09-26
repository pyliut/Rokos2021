# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:51:43 2021

@author: pyliu
"""

import numpy as np
import math
import scipy as sp

from calc_length import *

def calc_angle(a,b,c):
    """
    Calculate internal angle at b between points a -> b -> c
    i.e. angle abc (smaller than pi radians)
    Angle has units of radians

    Parameters
    ----------
    a : FLOAT, N-dim vector
        coords of point 1
    b : FLOAT, N-dim vector
        coords of point 2
    c : FLOAT, N-dim vector
        coords of point 3

    Returns
    -------
    FLOAT, scalar
        internal angle abc (radians)

    """
    #1) Convert to numpy for easy manipulation
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    #2) use trig
    ab = b - a
    bc = c - b
    len_ab = calc_length(a,b)
    len_bc = calc_length(b,c)
    cos_theta = np.sum( np.array(ab)*np.array(bc) ) / (len_ab * len_bc)
    return np.arccos(cos_theta)