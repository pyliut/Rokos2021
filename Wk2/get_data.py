# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:12:54 2021

@author: pyliu
"""

#import modules
import pandas as pd
import pymongo
from pymongo import MongoClient
from datetime import datetime

def get_data(address = "walmart_targeted"):
    """

    Parameters
    ----------
    address : STR
        Which collection to access in MongoDB. The default is "walmart_targeted".

    Returns
    -------
    df : Pandas DataFrame
        Data with no null or failed elements

    """
    #define my database and collection
    cluster = MongoClient("mongodb://127.0.0.1:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false")
    db = cluster["Rokos2021"]
    collection = db[address]
    
    #filter & store as pandas dataframe
    results = collection.find({"_meta.inserted_at": {"$gte": datetime(2021,2,20,9,0,0),
                                "$lte":datetime(2021,2,28,9,0,0)}, 
                                "origin": {"$ne": None}, 
                                "edge_id": {"$ne": None}, 
                                "target":{"$ne":None},
                                "succeeded":True})
    
    #convert to dataframe format
    df = pd.DataFrame(list(results))
    
    return df
