import json
import os
import sys
import copy
import tensorflow as tf
import pandas as pd
import numpy as np

import mfgm_imaterialist
import mfgm_imaterialist as iM
import mfgm_parameters as PARAMS
import mfgm_misc as MISC

import keras
from keras.utils import to_categorical
from keras import losses
from keras import layers
from keras import models
from keras.preprocessing import image
from keras import backend as backend
from keras.utils import metrics_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import load_model
from keras.callbacks import CSVLogger, EarlyStopping
import matplotlib.pyplot as plt

from keras.callbacks import Callback
from sklearn.metrics import accuracy_score

np.random.seed(8000)

def get_multitask_subset(data_type='training',
                         apparel_class='all',
                         tasks=('shoe:gender', 'shoe:age', 'shoe:color'),
                         randomize_missing_labels=True,
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
        print(f"### CURATED DATASET FOR APPAREL CLASS '{apparel_class}' ({data_type}) ###")


    if verbose > 1:
        print(f"Getting task-to-labels map...")

    dataset = None

    # task-to-labels map
    if data_type != 'training':
        rawdata = iM.get_dataset('training')

        """
                iM.import_rawdata(data_type='training',
                                                    delete_orphan_entries=False,
                                                    save_json=False,
                                                    attempt_downloading_images=False,
                                                    verbose=0)
                """

        task_to_labels = MISC.get_task_to_all_values_map(training_dataset=rawdata)
    else:
        dataset = iM.get_dataset(data_type)

        task_to_labels = MISC.get_task_to_all_values_map(training_dataset=dataset)

    # remove tasks not of this apparel class from task_to_labels
    # i.e. remove 'dress:length' or 'shoe:pump type' from the shoe tasks we're interested in
    all_tasks = MISC.get_tasks()
    for generic_task in all_tasks:
        if generic_task not in tasks:
            task_to_labels.pop(generic_task)

    number_of_labels = []
    for task in tasks:
        number_of_labels.append(len(task_to_labels[task]))

    # get dataset of only the relevant apparel class
    print(f"Importing {data_type} dataset...")
    if dataset is None: dataset = iM.get_dataset(data_type)

    if apparel_class in ('shoe', 'dress', 'outerwear', 'pants'):
        dataset = dataset[dataset['apparelClass'] == apparel_class]

    dataset = dataset[dataset['taskName'].apply(lambda x: x in tasks)]

    if verbose > 1:
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

    dataset['imageName'] = dataset['imageId'].apply(lambda x: f"{x}.jpg")

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

    # calculate tasks' fill rate
    fill_rate = dict()
    for task in tasks:
        fill_rate[task] = (1. - np.mean(dataset[task].isnull()))

    if verbose > 0:
        # how well filled are the tasks? how many labels do they have?
        task_info = pd.DataFrame(data=0, index=tasks, columns=["Fill Rate", "Amount of Labels"], dtype=int)

        task_info["Fill Rate"] = task_info["Fill Rate"].astype('float32')

        for task in tasks:
            task_info.loc[task, "Fill Rate"] = fill_rate[task] * 100.
            task_info.loc[task, "Amount of Labels"] = len(task_to_labels[task])

        task_info["Ratio"] = task_info["Fill Rate"] / task_info["Amount of Labels"]

        task_info.sort_values(by="Ratio", ascending=True, inplace=True)

        task_info["Fill Rate"] = task_info["Fill Rate"].apply(lambda x: f"{x:.2f}%")
        task_info["Ratio"] = task_info["Ratio"].apply(lambda x: f"{x:.2f}%")

        print("TASK FILL RATE AND AMOUNT OF LABELS:")
        print(task_info)

        print(f"\"Recommended\" order in which the 'tasks' variable should be: {list(reversed(task_info.index))}") #[{', '.join(reversed(task_info.index))}]

    # change labels from strings ('high') to ints (i.e. 3)
    if verbose > 1:
        print("Making labels integers...")

    def index_of(t, j, x):
        try:
            return task_to_labels[t].index(x)
        except ValueError:
            if randomize_missing_labels: # data_type == 'training' and
                return np.random.choice(range(number_of_labels[j]))

            return -1

    for i, task in enumerate(tasks):
        dataset[task] = dataset[task].apply(lambda x: index_of(task, i, x))

    if verbose > 1:
        print("FINAL FORM OF THE DATASET... (first 5 entries)")
        print(dataset.iloc[0:5, :])

    return dataset, number_of_labels, fill_rate


def get_untrained_multitask_model(
        image_size = 256,
        kernel_size = 4,
        maxpooling_size = 4,
        dropout_rate = 0.5,
        gamma = 0.95,
        tasks_arg = ('placeholder_1', 'placeholder_2', 'placeholder_3'),
        number_of_labels = (2,2,4),
        verbose = 1):
    """
    Input: image file
    Outputs:

    :return:
    """

    tasks = tasks_arg.copy()
    task_reformat = dict()

    if type(tasks) is list:
        for i, task_org in enumerate(tasks_arg):
            tasks[i] = tasks[i].replace(":", ".").replace(" ", "-")
            task_reformat[task_org] = tasks[i]

    if verbose > 0:
        print(f"Inside a model, tasks have been reformatted:\n{MISC.dictformat(task_reformat)}")

    image_shape = (image_size, image_size, 3)
    kernel_shape = (kernel_size, kernel_size)
    maxpooling_shape = (maxpooling_size, maxpooling_size)

    inputs = tf.keras.layers.Input(shape=image_shape, name='input')

    if verbose > 0:
        print("Defining common convolutional layers...")

    main_branch = tf.keras.layers.Conv2D(filters=8, kernel_size=kernel_shape, padding='same', activation='relu', name="conv_main_1")(inputs)
    main_branch = tf.keras.layers.MaxPooling2D(pool_size=maxpooling_shape, name="maxpool_main_1")(main_branch)
    main_branch = tf.keras.layers.Conv2D(filters=16, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu', name="conv_main_2")(main_branch)
    main_branch = tf.keras.layers.MaxPooling2D(pool_size=maxpooling_shape, name="maxpool_main_2")(main_branch)
    main_branch = tf.keras.layers.Conv2D(filters=32, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu', name="conv_main_3")(main_branch)
    main_branch = tf.keras.layers.Flatten()(main_branch)

    # preventing gradient explosion: normalize, tanh activation
    # main_branch = tf.keras.layers.BatchNormalization()(main_branch)
    main_branch = tf.keras.layers.Dense(128, activation='tanh', name="dense_main")(main_branch)

    # dropout to prevent overfitting
    main_branch = tf.keras.layers.Dropout(dropout_rate, name="dropout_main")(main_branch)

    if verbose > 0:
        print("Defining multi-task layers...")

    number_of_tasks = len(number_of_labels)
    task_branches = []
    weights_map = dict()
    loss_map = dict()

    for i in range(number_of_tasks):
        n = number_of_labels[i]
        lastlayer_activation = 'softmax'

        task_name = f'{tasks[i]}'

        task_branch = tf.keras.layers.Dense(16 * n, activation='relu', name=f"dense_{task_name}_1")(main_branch)
        task_branch = tf.keras.layers.Dense(8 * n, activation='relu', name=f"dense_{task_name}_2")(task_branch)
        # task_branch = tf.keras.layers.Dense(8 * n, activation='tanh', name=f"dense_{task_name}_3")(task_branch)
        # task_branch = tf.keras.layers.Dense(2 * n, activation='relu', name=f"dense_{task_name}_4")(task_branch)
        task_branch = tf.keras.layers.Dense(n, activation=lastlayer_activation, name=task_name)(task_branch)

        weights_map[task_name] = gamma**i

        loss_map[task_name] = losses.SparseCategoricalCrossentropy(from_logits=False, ignore_class=-1)

        task_branches.append(task_branch)

        if verbose > 0:
            print(f"Task {i+1}/{number_of_tasks} ({tasks[i]}) appended")

    # normalize weights
    weights_sum = np.sum(list(weights_map.values()))

    for key, value in weights_map.items():
        weights_map[key] = np.round(value / weights_sum, decimals=4)

    model = tf.keras.Model(inputs=inputs, outputs=task_branches)

    if verbose > 0: model.summary()

    # NOTE ABOUT SPARSE CATEGORICAL ACCURACY
    # It doesn't understand that -1 is a label to ignore,
    # so every time it's encountered it will count as a failed prediction.
    # If 75% of labels are valid (not -1), then the accuracy metric may never be higher than 75%
    # -> The accuracy metric needs to be normalized by the fill rate of that variable

    model.compile(optimizer='adam',
                  loss=loss_map,
                  loss_weights=weights_map,
                  metrics=['sparse_categorical_accuracy'])

    if verbose > 0:
        print(f"Weights (gamma**i, gamma = {gamma}):\n{MISC.dictformat(weights_map)}")
        print(f"Model successfully compiled")

    return model, task_reformat

def train_multitask_model(
        model,
        df_train,
        df_validation,
        tasks,
        number_of_labels,
        fill_rate_train,
        fill_rate_val,
        missing_labels_were_randomized,
        task_reformat,
        image_size = 256,
        batch_size = 32,
        epochs = 30,
        verbose = 1):
    """

    :return:
    """
    if verbose > 1:
        print(f"Creating training and validation generators...")

    classes_map = dict()
    for i, task in enumerate(tasks):
        classes_map[task] = list(range(number_of_labels[i]))

    image_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=0.25,
        height_shift_range=0.25,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=True,
    )

    train_generator = image_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory=PARAMS.IMAGES_filepath['training'],
        x_col='imageName',
        y_col=tasks,
        #classes=classes_map,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='multi_output'
    )

    validation_generator = image_datagen.flow_from_dataframe(
        dataframe=df_validation,
        directory=PARAMS.IMAGES_filepath['validation'],
        x_col='imageName',
        y_col=tasks,
        #classes=classes_map,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='multi_output'
    )

    if verbose > 1:
        for data_batch, labels_batch in train_generator:
            try:
                d = data_batch.shape
                l = labels_batch.shape
            except:
                d = f"{len(data_batch)}" #(i.e. {data_batch})
                l = f"{len(labels_batch)} (i.e. {labels_batch})"

            print(f"train_generator created with shape {d} and labels shape {l}")
            break

        for data_batch, labels_batch in validation_generator:
            try:
                d = data_batch.shape
                l = labels_batch.shape
            except:
                d = f"{len(data_batch)}"
                l = f"{len(labels_batch)} (i.e. {labels_batch})"

            print(f"validation_generator created with shape {d} and labels shape {l}")
            break

    # callbacks
    if verbose > 1:
        print("Defining callbacks...")

    # saving model
    folder_path = PARAMS.MISC_models_path

    try:
        os.mkdir(folder_path)
    except FileExistsError as error:
        if verbose > 1: print(f"{folder_path} folder already exists")
    else:
        if verbose > 1: print(f"{folder_path} folder created")

    file_path = f"{folder_path}/{PARAMS.MFGM_MULTITASK_MODEL_filename}"

    patience = np.max([epochs // 8, 6])

    if verbose > 0:
        print(f"Early stopping patience: {patience}")

    earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=patience,
                              verbose=1,
                              mode='auto',
                              restore_best_weights=True)

    traininglog_path = f"{file_path}_trainingLog.csv"
    csv_logger = CSVLogger(traininglog_path, separator=",", append=False)

    if len(tasks) > 1:
        print("""
NOTE:
As long as there is more than one task in the network,
any 'accuracy' metrics during training are severely underestimated,
due to them treating missing labels as a failed prediction.
This is accounted for in the training log CSV file, but not while the model is being trained.
(The adjustment is also done properly when the missing labels are replaced with a random one)
""")

    history = model.fit(train_generator,
                        steps_per_epoch=train_generator.samples // batch_size + 1,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.samples // batch_size + 1,
                        callbacks=[earlystop, csv_logger],
                        verbose=1
                        )

    if verbose > 0:
        print(f"Saving model to {file_path}...")

    model.save(file_path)

    if verbose > 0:
        print(f"{file_path} saved")

    # prep the training log CSV
    if verbose > 0:
        print("Prepping the training log...")

    traininglog_df = pd.read_csv(traininglog_path)

    # rename columns
    rename_dict = dict()

    for column in traininglog_df.columns:
        for task_original, task_reformatted in task_reformat.items():
            # objective:
            # rename 'shoe.color_sparse_categorical_accuracy' into 'shoe:color_accuracy'
            # rename 'val_shoe.color_sparse_categorical_accuracy' into 'val_shoe:color_accuracy'

            if task_reformatted in column:
                # we found that 'shoe.color' is in column
                new_column = column

                # 'shoe.color' to 'shoe:color'
                new_column = new_column.replace(task_reformatted, task_original)

                # self-explanatory
                new_column = new_column.replace("_sparse_categorical_accuracy", "_accuracy")

                # no need to act on "val_"

                rename_dict[column] = new_column

    traininglog_df.rename(columns=rename_dict, inplace=True)

    # adjust accuracy values
    for column in traininglog_df.columns:
        if "accuracy" in column:
            for task in tasks:
                if task in column:
                    # contains 'accuracy' in column name AND contains i.e. 'shoe:color' in column name
                    # => adjust

                    # we need: observed accuracy, fill rate, number of labels (in the "missing_labels_were_randomized" case)
                    observed_accuracy = traininglog_df[column]

                    # fill rate
                    # training data and validation data fill rate are potentially different
                    if "val" in column:
                        fill_rate = fill_rate_val[task]
                    else:
                        fill_rate = fill_rate_train[task]

                    if missing_labels_were_randomized:
                        # number of labels
                        n = number_of_labels[tasks.index(task)]

                        traininglog_df[column] = (observed_accuracy - (1. - fill_rate) / n) / fill_rate
                    else:
                        traininglog_df[column] = observed_accuracy / fill_rate

    # add "random guess" columns
    for task in tasks:
        column_name = f"{task}_randomguess"
        value = 1. / number_of_labels[tasks.index(task)]

        traininglog_df[column_name] = value

    traininglog_df.to_csv(traininglog_path)

    if verbose > 0:
        print("Prepping successful")

    if verbose > 0:
        print("Performing an example prediction...")

        for data_batch, labels_batch in validation_generator:
            print(model.predict(x=data_batch))
            break

    return history

