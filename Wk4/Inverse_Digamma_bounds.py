# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:55:29 2021

@author: pyliu
"""
import numpy as np
import scipy as sp

def Inverse_Digamma_bounds(k):
    """
    From Batir paper on Inverse Digamma bounds
    The lower bound is a good approximation for the Inverse Digamma function

    Parameters
    ----------
    k : FLOAT, scalar
        k ~ Digamma(lower_bound)

    Returns
    -------
    lower_bound : FLOAT, scalar
        Mathematical lower bound of the inverse digamma

    """
    lower_bound = 1 / np.log(1 + np.exp(-k))
    upper_bound = np.exp(k) + 0.5
    if lower_bound - upper_bound > lower_bound:
        print("ERROR: bounds are not close")
    return lower_bound