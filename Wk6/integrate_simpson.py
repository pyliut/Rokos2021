# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:07:17 2021

@author: pyliu
"""
import numpy as np

def integrate_simpson(integrand,spacing):
    """
    Numerical integration using Simpson's rule

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
    coeff = []
    for i in range(len(integrand)):
        coeff.append(2**(1 + i%2))
    coeff[0] = 1
    coeff[-1] = 1
    
    integrand = np.multiply(integrand, coeff)
    
    return np.sum(integrand) * spacing / 3