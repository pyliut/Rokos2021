# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:58:51 2021

@author: pyliu
"""
import numpy as np
import scipy as sp

def Inverse_Digamma_newton(y):
    """
    An alternative to Inverse_Digamma_bounds
    This uses Newton's method to numerically solve Inverse Digamma
    Sourced from: https://www.programcreek.com/python/example/76869/scipy.special.digamma
    
    Parameters
    ----------
    y : FLOAT, scalar
        y ~ Digamma(output)

    Raises
    ------
    RuntimeError
        Raised when fsolve runs for too long

    Returns
    -------
    FLOAT, scalar
        output = Inverse_Digamma(y)

    """
    _em = 0.5772156649015328606065120
    func = lambda x: sp.special.digamma(x) - y
    if y > -0.125:
        x0 = np.exp(y) + 0.5
        if y < 10:
            # Some experimentation shows that newton reliably converges
            # must faster than fsolve in this y range.  For larger y,
            # newton sometimes fails to converge.
            value = sp.optimize.newton(func, x0, tol=1e-10)
            return value
    elif y > -3:
        x0 = np.exp(y/2.332) + 0.08661
    else:
        x0 = 1.0 / (-y - _em)

    value, info, ier, mesg = sp.optimize.fsolve(func, x0, xtol=1e-11,
                                             full_output=True)
    if ier != 1:
        raise RuntimeError("_digammainv: fsolve failed, y = %r" % y)

    return value[0]