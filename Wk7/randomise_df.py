# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:58:54 2021

@author: pyliu
"""
import pandas as pd

def randomise_df(df):
    """
    Shuffle rows of df

    Parameters
    ----------
    df : pandas df
        input df
    Returns
    -------
    df : pandas df
        output df

    """
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df