# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:25:40 2022

@author: Michelle
"""

# Importing packages
import pandas as pd
import numpy as np
import json

# LABEL MAP
def label_map_csv():
    with open('fgvc4_iMat.label_map.json','r') as f:
        data = json.loads(f.read())
    label_map = pd.json_normalize(data, record_path =['labelInfo'])
    label_map.to_csv('filter_label_map.csv')
    return

def task_map_csv():
    with open('fgvc4_iMat.task_map.json','r') as f:
        data = json.loads(f.read())
    task_map = pd.json_normalize(data, record_path =['taskInfo'])
    tasks = task_map[['taskName']]
    apparelTask = []
    typeTask = []
    for i in range(len(tasks)):
        elem = tasks.loc[i, 'taskName']
        splt = elem.split(':')
        apparelTask.append(splt[0])
        typeTask.append(splt[1])
    
    task_map['apparelTask'] = apparelTask
    task_map['typeTask']    = typeTask
    task_map = task_map[['taskId', 'apparelTask', 'typeTask']]
    task_map.to_csv('filter_task_map.csv')
    return

def test_image_csv():
    with open('fgvc4_iMat.test.image','r') as f:
        data = json.loads(f.read())
    test_image = pd.json_normalize(data, record_path =['images']) 
    test_image.to_csv('test_image.csv')
    return

def train_data_csv():
    with open('fgvc4_iMat.train.data','r') as f:
        data = json.loads(f.read())
    train_data1 = pd.json_normalize(data, record_path =['images'])
    train_data2 = pd.json_normalize(data, record_path = ['annotations']) 
    train_data = pd.merge(train_data1, train_data2, how = 'left', left_on = 'imageId', right_on = 'imageId')
    train_data.to_csv('train_data.csv')
    return

def validation_data_csv():
    with open('fgvc4_iMat.validation.data','r') as f:
        data = json.loads(f.read())
    val_data1 = pd.json_normalize(data, record_path =['images'])
    val_data2 = pd.json_normalize(data, record_path = ['annotations']) 
    val_data = pd.merge(val_data1, val_data2, how = 'left', left_on = 'imageId', right_on = 'imageId')
    val_data.to_csv('validation_data.csv')
    return

#label_map_csv()
#task_map_csv()
#test_image_csv()
#train_data_csv()
#validation_data_csv()