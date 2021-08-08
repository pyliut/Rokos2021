# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:43:27 2021

@author: pyliu
"""
from copy import copy
def integrate_pdf(pdf, spacing = 0.01):
    """
    Integrate discrete pdf into cdf

    Parameters
    ----------
    pdf : FLOAT, vector
        discrete pdf

    Returns
    -------
    pdf : FLOAT, vector
        discrete cdf

    """
    cdf = copy(pdf)
    for i in range(1,len(pdf)):
        cdf[i] = cdf[i] + cdf[i-1]
    return cdf*spacing