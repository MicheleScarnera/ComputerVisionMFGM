import json
import os
import tensorflow as tf
import pandas as pd

import mfgm_imaterialist
import mfgm_parameters as PARAMS

from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import load_model
from keras.callbacks import CSVLogger, EarlyStopping
import matplotlib.pyplot as plt

def get_untrained_apparel_model(
        image_size = 256,
        kernel_size = 3,
        maxpooling_size = 2,
        dropout_rate = 0.5,
        verbose = 1):

    kernel_square = (kernel_size, kernel_size)
    maxpooling_square = (maxpooling_size, maxpooling_size)

    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_square, padding='same', activation='relu',
                            input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D(maxpooling_square))
    model.add(layers.Conv2D(64, kernel_square, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(maxpooling_square))
    model.add(layers.Conv2D(128, kernel_square, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(maxpooling_square))
    model.add(layers.Conv2D(128, kernel_square, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(maxpooling_square))
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    if verbose > 0: model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=1e-4),
                  metrics=['acc'])

    return model

def train_apparel_model(
        model,
        image_size = 256,
        batch_size = 4,
        epochs = 5,
        verbose = 2):

    if verbose > 1:
        print(f"Creating {PARAMS.IMAGES_filepath['training']} and {PARAMS.IMAGES_filepath['validation']} generators...")

    train_datagen = ImageDataGenerator(rescale = 1. / 255)

    train_generator = train_datagen.flow_from_directory(
        PARAMS.IMAGES_filepath['training'],
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = validation_datagen.flow_from_directory(
        PARAMS.IMAGES_filepath['validation'],
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    if verbose > 1:
        for data_batch, labels_batch in train_generator:
            print(f"train_generator created with shape {data_batch.shape} and labels shape {labels_batch.shape}")
            break
        
        for data_batch, labels_batch in validation_generator:
            print(f"validation_generator created with shape {data_batch.shape} and labels shape {labels_batch.shape}")
            break

    #callbacks
    if verbose > 1:
        print("Defining callbacks...")

    earlystop = EarlyStopping(monitor='val_acc',
                              min_delta=0.001,
                              patience=6,
                              verbose=1,
                              mode='auto')

    csv_logger = CSVLogger('training.log', separator=",", append=False)

    history = model.fit(train_generator,
                        steps_per_epoch=train_generator.samples / batch_size,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.samples / batch_size,
                        callbacks=[earlystop, csv_logger]
                        )

    #saving model
    folder_path = PARAMS.MISC_models_path
    file_path = f"{folder_path}/{PARAMS.CV_MODELS_APPAREL_filename}"

    if verbose > 0:
        print(f"Saving apparel model to {file_path}...")

    try:
        os.mkdir(folder_path)
    except FileExistsError as error:
        if verbose > 1: print(f"{folder_path} folder already exists")
    else:
        if verbose > 1: print(f"{folder_path} folder created")

    model.save(file_path)

    if verbose > 0:
        print(f"{file_path} saved")

    return history


import_imaterialist.get_dataset()
import_imaterialist.get_dataset('validation')

apparel_model = get_untrained_apparel_model()
train_apparel_model(apparel_model)
