# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:05:28 2021

@author: pyliu
"""
import numpy as np
def integrate(integrand,spacing):
    """
    Numerical integration using rectangles

    Parameters
    ----------
    integrand : FLOAT, vector
        Values of integrand in the range of integration
    spacing : FLOAT, scalar
        Width of integrating strips

    Returns
    -------
    FLOAT, scalar
        Integrated value

    """
    return np.sum(integrand)*spacing