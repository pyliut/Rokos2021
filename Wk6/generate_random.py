# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 09:51:07 2021

@author: pyliu
"""
import numpy as np
import math
import scipy as sp

from integrate_pdf import *

def generate_random(t, pdf, n_random = 50):
    """
    Generate random values for a given PDF

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    cdf : TYPE
        DESCRIPTION.
    n_random : TYPE, optional
        DESCRIPTION. The default is 50.

    Returns
    -------
    gen : TYPE
        DESCRIPTION.

    """
    gen_uniform = sp.stats.uniform.rvs(loc=0, scale=1, size=n_random, random_state=None)
    gen = []
    
    cdf = integrate_pdf(pdf, spacing = t[1] - t[0])

    for i in range(len(gen_uniform)):
        diff = cdf - gen_uniform[i]
        ind = np.argmin(np.abs(diff))
        gen.append(t[ind])
    
    return gen
    