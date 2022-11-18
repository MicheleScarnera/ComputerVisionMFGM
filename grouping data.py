# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 10:05:31 2022

@author: Michelle
"""

import pandas as pd
import numpy as np
import json

attribute = 'gender'
attribute = 'color'
attribute = 'age'
attribute = 'material'

data = pd.read_csv('trainingData.csv')
    
threshold = 1/100

def group_labels(attribute, threshold):
    apparel = ['shoe', 'dress', 'pants', 'outerwear']   
    
    # get labelids only belonging to the right class
    taskIds = []
    for j in range(len(apparel)):
        b = data.loc[data['taskName'] == apparel[j]+':'+attribute, 'taskId'].unique()
        taskIds = [*taskIds, *b]
    
    # select subset of data only containing these task-ids
    subset = data[data['taskId'].isin(taskIds)]
    
    # count how often each label id occurs
    a= subset.groupby('labelId').count()
    
    # calculate ratio
    df = pd.DataFrame({'labelId': a.index, 'counts': a['imageId']}).reset_index(drop=True)
    df['ratio'] = df['counts']/ sum(df['counts'])
    
    # group every label that is used in less than 1% (threshold) of the cases
    to_replace = list(df.loc[df['ratio']<threshold, 'labelId'])
    
    # set this labelId to 999
    data.loc[data['labelId'].isin(to_replace), 'labelId'] = 999

    return data