# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:15:40 2022

@author: Michelle
"""

#%% Loading packages

import pandas as pd
import numpy as np
import requests
import os
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping
import matplotlib.pyplot as plt


#%% FUNCTIONS

def get_subset_shoe(data, task_id1, task_id2):
   
    shoe_task1 = data.loc[data['taskId']==task_id1,:]
    shoe_task2 = data.loc[data['taskId']==task_id2,:]

    labels = pd.merge(shoe_task1, shoe_task2, how = 'outer', right_on = 'imageId', left_on = 'imageId',)

    # Combining the information of both labels (for age and gender)
    labels['labelId_xy']=pd.Series(zip(labels["labelId_x"], labels["labelId_y"])).map(list)
    
    # Adding information on how many shoes have both labels
    labels.loc[:, 'both'] = 0
    labels.loc[(pd.isnull(labels['labelId_x']) == False) & (pd.isnull(labels['labelId_y'])== False),'both'] = 1
    labels = labels.loc[:, ['imageId', 'labelId_x', 'labelId_y', 'labelId_xy', 'both', 'url_x', 'url_y']]
    # Adding information on the filename for later use
    labels['Filenames'] = labels.loc[:,'imageId'].astype(str)+'.jpg'
    
    return labels



def download_images(img_ids, data):
    
    for i in range(len(img_ids)):
        #print(i)
        img_id = img_ids[i]
        url_list = data.loc[data['imageId']==img_id, 'url_x']
        
        if pd.isnull(url_list.iloc[0]) == True:
            url_list = data.loc[data['imageId']==img_id, 'url_y']
            
        url_list = url_list.iloc[0].split(',') 
        name = str(img_id)
    
        for j in range(len(url_list)):
            img_url = url_list[j].strip('\'[]')
    
            try:
                res = requests.get(img_url,stream=True, timeout=1)
                
            except requests.exceptions.ConnectionError:
                continue
            except requests.exceptions.InvalidSchema:
                continue
            
            except requests.exceptions.ReadTimeout:
                continue
            except OSError:
                continue
            except ValueError:
                continue
            
            try:
                if res.ok and '<' not in res.text[0:20] and len(res.content) > 5000:
                    fp = open(f"{name}.jpg", 'wb')
                    fp.write(res.content)
                    fp.close()
                    break

            except OSError:
                continue
            except ValueError:
                continue
    return 

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',      # first convolutional layer
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))                      # first max pooling
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))     # second convolutional layer
    model.add(layers.MaxPooling2D((2, 2)))                      # second pooling
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))    # third convolutional layer
    model.add(layers.MaxPooling2D((2, 2)))                      # third pooling
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))    # fourth convolutional layer
    model.add(layers.MaxPooling2D((2, 2)))                      # fourth pooling
    model.add(layers.Flatten())                                 # flattening it out
    model.add(layers.Dense(512, activation='relu'))             # first dense layer
    model.add(layers.Dense(n_outputs, activation='sigmoid'))            # last dense layer
    
    print(model.summary())

    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
    
    return 

def run_model(train, base_dir_train, validation, base_dir_val, columns, batch, \
              image_size, epoch_steps, epoch, val_steps, n_outputs):

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',      # first convolutional layer
                            input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))                      # first max pooling
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))     # second convolutional layer
    model.add(layers.MaxPooling2D((2, 2)))                      # second pooling
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))    # third convolutional layer
    model.add(layers.MaxPooling2D((2, 2)))                      # third pooling
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))    # fourth convolutional layer
    model.add(layers.MaxPooling2D((2, 2)))                      # fourth pooling
    model.add(layers.Flatten())                                 # flattening it out
    model.add(layers.Dense(512, activation='relu'))             # first dense layer
    model.add(layers.Dense(n_outputs, activation='sigmoid'))            # last dense layer
    
    print(model.summary())
    
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
    
    
    # Rescale all images by 1./255
    train_datagen = ImageDataGenerator(rescale=1./255)
    
    
    train_generator=train_datagen.flow_from_dataframe(dataframe=train, 
                                                directory=base_dir_train,
                                                x_col="Filenames",
                                                y_col=columns,
                                                batch_size=batch,
                                                class_mode="raw",
                                                target_size=(image_size,image_size),
                                                validate_filenames=False)
    
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator=validation_datagen.flow_from_dataframe(dataframe=validation, 
                                                directory=base_dir_val,
                                                x_col="Filenames",
                                                y_col=columns,
                                                batch_size=batch,
                                                class_mode="raw",
                                                target_size=(image_size,image_size),
                                                validate_filenames=False)
    
    history = model.fit(
          train_generator,
          steps_per_epoch=epoch_steps,
          epochs=epoch,
          validation_data=validation_generator,
          validation_steps=val_steps,
          #callbacks=[earlystop, csv_logger]
    )
    
    return 
#%% 

base_dir_train = './train'
base_dir_val = './validation'

# Loading the data
train_data = pd.read_csv('train_data.csv')
validation_data = pd.read_csv('validation_data.csv')

# Selecting the task_ids (1 = gender, 2 = age)
task_id1 = 1
task_id2 = 2

# Only selecting the data that is relevant for shoes
train = get_subset_shoe(train_data, task_id1, task_id2)
validation = get_subset_shoe(validation_data, task_id1, task_id2)

# Downloading images
# download_images(train['imageId'].unique(), train)
# download_images(validation['imageId'].unique(), validation)

# editing our train and validation set such that it only contains images 
# that we have actually downloaded
d_train =os.listdir(base_dir_train)
train_new = train.loc[train['Filenames'].isin(d_train)]

d_val = os.listdir(base_dir_val)
val_new = validation.loc[validation['Filenames'].isin(d_val)]

# Inputs for the model
batch = 20
image_size = 150
epoch_steps = 10
epoch = 30
val_steps = 5
n_outputs = 2
columns = ['labelId_x', 'labelId_y']
run_model(train_new, base_dir_train, val_new, base_dir_val, columns, batch, image_size, epoch_steps, epoch, val_steps, n_outputs)


