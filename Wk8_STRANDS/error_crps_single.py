# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:40:29 2021

@author: pyliu
"""
import numpy as np
import pandas as pd

from integrate import *
from integrate_trapezium import *
from integrate_simpson import *

def error_crps_single(t_new, cdf, t_test, method = "simpson"):
    """
    Calculate Continuous Ranked Probability Score (CRPS)
    for a single new observations

    Parameters
    ----------
    t_new : FLOAT, scalar
        out-of-sample observations
    cdf : FLOAT, vector
        cdf values of predicted distribution (dependent variable)
    t_test : FLOAT, vector
        independent variable of cdf
    method : STR
        type of numerical integration method
        Options are: "rectangle", "trapezium", "simpson".
        The default is "simpson".

    Returns
    -------
    crps : FLOAT, scalar
        CRPS score for new observation

    """
    #integrand for crps integral
    integrand1 = cdf**2
    integrand2 = (cdf - 1)**2
    
    #constants for integration strip_width and rounding precision
    t_step = t_test[1] - t_test[0]
    precision = -int( np.log10(t_step) )
    #2a) round observation
    t_round = round(t_new, precision)

    #2b) Find limits of integration
    lower_lim1 = 0
    ind = np.where(t_test == t_round)[0]
    if len(ind) > 1:
        upper_lim1 = int(ind[0])
    elif len(ind) == 1:
        upper_lim1 = int(ind)

    #3) Perform integration
    if len(ind) != 0:
        if method == "rectangle":
            result1 = integrate(integrand1[lower_lim1:upper_lim1],t_step)
            result2 = integrate(integrand2[upper_lim1:],t_step)
        elif method == "trapezium":
            result1 = integrate_trapezium(integrand1[lower_lim1:upper_lim1],t_step)
            result2 = integrate_trapezium(integrand2[upper_lim1:],t_step)
        elif method == "simpson":
            result1 = integrate_simpson(integrand1[lower_lim1:upper_lim1],t_step)
            result2 = integrate_simpson(integrand2[upper_lim1:],t_step)
        else:
            raise ValueError("method is not valid. Options are: rectangle, trapezium, simpson")
    else:
        raise ValueError("t_new is out of range")
    return result1 + result2