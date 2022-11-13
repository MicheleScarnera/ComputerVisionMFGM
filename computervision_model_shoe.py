import json
import os
import copy
import tensorflow as tf
import pandas as pd
import numpy as np

import import_imaterialist as iM
import computervision_parameters as PARAMS
import computervision_misc as MISC

import keras
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import load_model
from keras.callbacks import CSVLogger, EarlyStopping
import matplotlib.pyplot as plt

def get_untrained_shoe_to_gender_model(
        image_size = 256,
        kernel_size = 3,
        maxpooling_size = 2,
        dropout_rate = 0.5,
        verbose = 1):
    """
    Input: image file
    Outputs: gender 2-vector (+ 512-vector?)

    :return:
    """

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
    model.add(layers.Dense(2, activation='softmax'))

    if verbose > 0: model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=1e-4),
                  metrics=['acc'])

    return model

def train_shoe_to_gender_model(
        model,
        image_size = 256,
        batch_size = 4,
        epochs = 5,
        verbose = 2):
    """

    :param model:
    :param image_size:
    :param batch_size:
    :param epochs:
    :param verbose:
    :return:
    """

    if verbose > 1:
        print(f"Creating {PARAMS.IMAGES_shoepath['training']} and {PARAMS.IMAGES_shoepath['validation']} generators...")

    train_datagen = ImageDataGenerator(rescale = 1. / 255)

    df_train = iM.get_dataset('training')

    df_train = df_train[df_train['taskName'] == 'shoe:gender']

    #print(df_train[['imageName', 'labelName']])
    #print(df_train['imageName'].shape)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=PARAMS.IMAGES_shoepath['training'],
        x_col='imageName',
        y_col='labelName',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    df_validation = iM.get_dataset('validation')

    df_validation = df_validation[df_validation['taskName'] == 'shoe:gender']

    #print(df_validation[['imageName', 'labelName']])
    #print(df_validation['imageName'].shape)

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=df_validation,
        directory=PARAMS.IMAGES_shoepath['validation'],
        x_col='imageName',
        y_col='labelName',
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

    # callbacks
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

    # saving model
    folder_path = PARAMS.MISC_models_path
    file_path = f"{folder_path}/{PARAMS.CV_MODELS_SHOE_TO_GENDER_filename}"

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

def get_untrained_shoe_autoencoder(
        image_size = 256,
        kernel_size = 3,
        maxpooling_size = 2,
        dropout_rate = 0.5,
        verbose = 1):
    """



    :param image_size:
    :param kernel_size:
    :param maxpooling_size:
    :param dropout_rate:
    :param verbose:
    :return: Returns a 2-tuple, containing (model, encoder)
    """

    image_square = (image_size, image_size, 3)
    kernel_square = (kernel_size, kernel_size)
    maxpooling_square = (maxpooling_size, maxpooling_size)

    input_img = keras.Input(shape=image_square)

    x = layers.Conv2D(64, kernel_square, activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D(maxpooling_square, padding='same')(x)
    x = layers.Conv2D(32, kernel_square, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(maxpooling_square, padding='same')(x)
    x = layers.Conv2D(32, kernel_square, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D(maxpooling_square, padding='same', name='encoder')(x)

    # at this point the representation is

    x = layers.Conv2D(32, kernel_square, activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D(maxpooling_square)(x)
    x = layers.Conv2D(32, kernel_square, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(maxpooling_square)(x)
    x = layers.Conv2D(64, kernel_square, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(maxpooling_square)(x)
    decoded = layers.Conv2D(3, kernel_square, activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded)

    if verbose > 0: autoencoder.summary()

    autoencoder.compile(
        optimizer=optimizers.RMSprop(learning_rate=1e-4),
        loss='binary_crossentropy')

    return autoencoder, encoded

def train_shoe_autoencoder(
        model,
        image_size = 256,
        batch_size = 4,
        epochs = 5,
        verbose = 2):
    """

    :return:
    """
    if verbose > 1:
        print(f"Creating {PARAMS.IMAGES_shoepath['training']} and {PARAMS.IMAGES_shoepath['validation']} generators...")

    train_datagen = ImageDataGenerator(rescale = 1. / 255)

    df_train = iM.get_dataset('training')

    df_train = df_train[df_train['apparelClass'] == 'shoe']

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=PARAMS.IMAGES_shoepath['training'],
        x_col='imageName',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='input'
    )

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    df_validation = iM.get_dataset('validation')

    df_validation = df_validation[df_validation['apparelClass'] == 'shoe']

    # print(df_validation[['imageName', 'labelName']])
    # print(df_validation['imageName'].shape)

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=df_validation,
        directory=PARAMS.IMAGES_shoepath['validation'],
        x_col='imageName',
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='input'
    )

    if verbose > 1:
        for data_batch, labels_batch in train_generator:
            print(f"train_generator created with shape {data_batch.shape} and labels shape {labels_batch.shape}")
            break

        for data_batch, labels_batch in validation_generator:
            print(f"validation_generator created with shape {data_batch.shape} and labels shape {labels_batch.shape}")
            break

    # callbacks
    if verbose > 1:
        print("Defining callbacks...")

    earlystop = EarlyStopping(monitor='val_loss',
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

    # saving model
    folder_path = PARAMS.MISC_models_path
    file_path = f"{folder_path}/{PARAMS.CV_MODELS_SHOE_AUTOENCODER_filename}"

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

def get_shoe_subset(data_type='training',
                    apparel_class='shoe',
                    tasks=('shoe:gender', 'shoe:age', 'shoe:color'),
                    verbose=2):
    """
    NOTE: this can be generalized for other apparel classes. All of the variables that would need to be changed
    (i.e. become arguments of a more general function) have been defined right after this comment

    :return: dataset, number_of_labels
    """
    # TO-BE-GENERALIZED VARIABLES

    # all shoe tasks: ['shoe:gender', 'shoe:age', 'shoe:color', 'shoe:up height', 'shoe:type', 'shoe:closure type', 'shoe:toe shape', 'shoe:heel type', 'shoe:decoration', 'shoe:flat type', 'shoe:material', 'shoe:back counter type', 'shoe:pump type', 'shoe:boot type']
    # 'shoe:boot type' and 'shoe:pump type' have been removed since they only have one label in the training set

    # TO-BE-GENERALIZED VARIABLES OVER

    if verbose > 0:
        print(f"### CURATED DATASET FOR APPAREL CLASS '{apparel_class}'###")

    # task-to-labels map
    rawdata = iM.import_rawdata(data_type='training', delete_orphan_entries=False, save_json=False, verbose=0)
    task_to_labels = MISC.get_task_to_all_values_map(training_dataset=rawdata)

    # remove tasks not of this apparel class from task_to_labels
    # i.e. remove 'dress:length' or 'shoe:pump type' from the shoe tasks we're interested in
    all_tasks = MISC.get_tasks()
    for generic_task in all_tasks:
        if generic_task not in tasks:
            task_to_labels.pop(generic_task)

    number_of_labels = []
    for task,labels in task_to_labels.items():
        number_of_labels.append(len(labels))

    # get dataset of only the relevant apparel class
    dataset = iM.get_dataset(data_type)
    dataset = dataset[dataset['apparelClass'] == apparel_class]

    if verbose > 0:
        print("PRINTING DATASET... (first 5 entries)")
        print(dataset.iloc[0:5,:])

        arbitraryimage_id = dataset.iloc[len(dataset) // 2]['imageId']
        print("FINDING FEATURES OF ONE IMAGE...")
        print(dataset[dataset['imageId'] == arbitraryimage_id])

    # create columns for each task, and assign label when found
    # the result will be so that, for example:
    # each row will have columns ['shoe:gender', 'shoe:age', ... , 'shoe:color', 'shoe:type']
    # with values ['men', 'adult', ... , 'blue', 'sneaker']
    # (most images I've seen rarely have this many fields filled)

    if verbose > 0:
        print("Reformatting dataset...")

    dataset.drop_duplicates(subset=['imageId', 'taskName'], inplace=True)

    dataset = dataset.set_index(['imageId', 'taskName'], drop=False, verify_integrity=True)['labelName'].unstack().reset_index()

    if verbose > 0:
        print("Dataset reformatted")

    if verbose > 1:
        print("PRINTING REFORMATTED DATASET... (first 5 entries)")
        print(dataset.iloc[0:5,:])
        print("FINDING FEATURES OF ONE IMAGE AFTER REFORMATTING...")

        pd.set_option('display.max_columns', None)
        print(dataset[dataset['imageId'] == arbitraryimage_id])
        pd.reset_option('display.max_columns')

        #print(dataset[dataset["imageId"] == "4"])

        # how well filled are the tasks? how many labels do they have?
        task_info = pd.DataFrame(data=0, index=tasks, columns=["Fill Rate", "Amount of Labels"], dtype=int)

        task_info["Fill Rate"] = task_info["Fill Rate"].astype('float32')

        for task in tasks:
            task_info.loc[task, "Fill Rate"] = (1. - np.mean(dataset[task].isnull())) * 100.
            task_info.loc[task, "Amount of Labels"] = len(task_to_labels[task])

        task_info["Ratio"] = task_info["Fill Rate"] / task_info["Amount of Labels"]

        task_info.sort_values(by="Ratio", ascending=True, inplace=True)

        task_info["Fill Rate"] = task_info["Fill Rate"].apply(lambda x: f"{x:.2f}%")
        task_info["Ratio"] = task_info["Ratio"].apply(lambda x: f"{x:.2f}%")

        print("TASK FILL RATE AND AMOUNT OF LABELS:")
        print(task_info)

    return dataset, number_of_labels

def get_untrained_multitask_shoe_model(
        image_size = 256,
        kernel_size = 3,
        maxpooling_size = 4,
        dropout_rate = 0.5,
        gamma = 0.5,
        number_of_labels = [2,2,4],
        verbose = 1):
    """
    Input: image file
    Outputs:

    :return:
    """

    image_shape = (image_size, image_size, 3)
    kernel_shape = (kernel_size, kernel_size)
    maxpooling_shape = (maxpooling_size, maxpooling_size)

    inputs = tf.keras.layers.Input(shape=image_shape, name='input')

    if verbose > 0:
        print("Defining common convolutional layers...")

    main_branch = tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_shape, padding='same', name="conv_main_1")(inputs)
    main_branch = tf.keras.layers.MaxPooling2D(pool_size=maxpooling_shape, name="maxpool_main_1")(main_branch)
    main_branch = tf.keras.layers.Conv2D(filters=64, kernel_size=(kernel_size, kernel_size), padding='same', name="conv_main_2")(main_branch)
    main_branch = tf.keras.layers.MaxPooling2D(pool_size=maxpooling_shape, name="maxpool_main_2")(main_branch)
    main_branch = tf.keras.layers.Conv2D(filters=128, kernel_size=(kernel_size, kernel_size), padding='same', name="conv_main_3")(main_branch)
    main_branch = tf.keras.layers.Flatten()(main_branch)
    main_branch = tf.keras.layers.Dense(512, activation='relu', name="dense_main")(main_branch)
    main_branch = tf.keras.layers.Dropout(dropout_rate, name="dropout_main")(main_branch)

    if verbose > 0:
        print("Defining multi-task layers...")

    number_of_tasks = len(number_of_labels)
    task_branches = []
    loss_map = dict()
    weights_map = dict()

    for i in range(number_of_tasks):
        n = number_of_labels[i]
        task_name = f'task_{i+1}_output'

        task_branch = tf.keras.layers.Dense(64 * n, activation='relu', name=f"dense_task{i+1}_1")(main_branch)
        task_branch = tf.keras.layers.Dense(32 * n, activation='relu', name=f"dense_task{i+1}_2")(task_branch)
        task_branch = tf.keras.layers.Dense(16 * n, activation='relu', name=f"dense_task{i+1}_3")(task_branch)
        task_branch = tf.keras.layers.Dense(8 * n, activation='relu', name=f"dense_task{i+1}_4")(task_branch)
        task_branch = tf.keras.layers.Dense(n, activation='softmax', name=task_name)(task_branch)

        if n > 2:
            loss_map[task_name] = 'categorical_crossentropy'
        else:
            loss_map[task_name] = 'binary_crossentropy'

        weights_map[task_name] = gamma**i

        task_branches.append(task_branch)

        if verbose > 0:
            print(f"Task {i+1}/{number_of_tasks} appended")

    model = tf.keras.Model(inputs=inputs, outputs=task_branches)

    if verbose > 0: model.summary()

    model.compile(optimizer='adam',
                  loss=loss_map,
                  loss_weights=weights_map,
                  metrics=['accuracy'])

    if verbose > 0:
        print(f"Model successfully compiled")

    return model


#iM.get_dataset()
#iM.get_dataset('validation')

tasks=['shoe:gender', 'shoe:age', 'shoe:color', 'shoe:up height', 'shoe:type', 'shoe:closure type',
    'shoe:toe shape', 'shoe:heel type', 'shoe:decoration', 'shoe:flat type', 'shoe:material',
    'shoe:back counter type']

dataset, number_of_labels = get_shoe_subset(apparel_class='shoe', tasks=tasks)
get_untrained_multitask_shoe_model(number_of_labels=number_of_labels)