import json
import os
import tensorflow as tf
import pandas as pd

import import_imaterialist

from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import load_model
from keras.callbacks import CSVLogger, EarlyStopping
import matplotlib.pyplot as plt


def get_untrained_apparel_model(image_size = 256, kernel_size = 3, maxpooling_size = 4, verbose = 1):
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
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    if verbose > 0: model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=1e-4),
                  metrics=['acc'])

    return model

def train_apparel_model(model, batch_size = 16, epochs = 5, verbose = 2):
    """
    earlystop = EarlyStopping(monitor='val_acc',
                              min_delta=0.001,
                              patience=6,
                              verbose=1,
                              mode='auto')

    csv_logger = CSVLogger('training.log', separator=",", append=False)
    """

    if verbose > 1:
        print(f"Opening {import_imaterialist.EXPORTEDJSON_trainingPath}...")

    with open(import_imaterialist.EXPORTEDJSON_trainingPath) as read_content:
        jsonfile = json.load(read_content)

    if verbose > 1:
        print(f"{import_imaterialist.EXPORTEDJSON_trainingPath} opened...")

    if verbose > 1:
        print("Getting train_images...")

    #print(jsonfile['imageData'])
    train_images = tf.convert_to_tensor(list(jsonfile['imageData'].values()))
    if verbose > 1:
        print(f"train_images obtained with shape {train_images.shape}")

    if verbose > 2:
        print(train_images)

    if verbose > 1:
        print("Getting train_labels...")
    train_labels = to_categorical(list(jsonfile['apparel_class'].values()))
    if verbose > 1:
        print(f"train_labels obtained with {train_labels.shape[1]} classes")

    if verbose > 2:
        print(train_labels)

    jsonfile = None

    model.fit(train_images, train_labels, epochs = epochs, batch_size = batch_size) #, callbacks = [earlystop, csv_logger]


apparel_model = get_untrained_apparel_model()

#dataset = import_imaterialist.get_dataset()

train_apparel_model(apparel_model)