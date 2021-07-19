# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 15:43:27 2021

@author: pyliu
"""

def integrate_pdf(pdf):
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
    for i in range(1,len(pdf)):
        pdf[i] = pdf[i] + pdf[i-1]
    return pdf