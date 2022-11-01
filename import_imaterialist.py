import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as plt
import imageio.v3 as iio
import requests
import json
from io import BytesIO
import time

EXPORTEDJSON_trainingPath = "trainingData.json"
EXPORTEDJSON_validationPath = "validationData.json"

EXPORTEDJSON_filepath = {'training': EXPORTEDJSON_trainingPath, 'validation': EXPORTEDJSON_validationPath}

RAWDATA_trainingPath = "iMaterialist/fgvc4_iMat.train.data.json"
RAWDATA_validationPath = "iMaterialist/fgvc4_iMat.validation.data.json"

RAWDATA_filepath = {'training': RAWDATA_trainingPath, 'validation': RAWDATA_validationPath}

DATASET_numberOfEntries = 200

def timeformat(secs):
    if type(secs) is not int: secs = int(secs)
    # Formats an integer secs into a HH:MM:SS format.
    return f"{str(secs // 3600).zfill(2)}:{str((secs // 60) % 60).zfill(2)}:{str(secs % 60).zfill(2)}"


def get_task_map() -> dict[int, str]:
    taskMapPath = "iMaterialist/fgvc4_iMat.task_map.json"

    with open(taskMapPath, 'r') as f:
        taskMap_raw = json.loads(f.read())

    mappings = taskMap_raw['taskInfo']
    result = dict()

    for l in mappings:
        result[int(l['taskId'])] = l['taskName']

    return result


def get_apparel_class_map(task_map=None) -> dict[int, int]:
    if (task_map == None):
        task_map = get_task_map()

    string_to_int = {
        'outerwear': 0,
        'dress': 1,
        'pants': 2,
        'shoe': 3
    }

    result = dict()
    for key, value in task_map.items():
        result[key] = string_to_int[value.split(":")[0]]

    return result


def get_image_file(url_list, image_size = 256, verbose = 0):
    if type(url_list) is str:
        url_list = [url_list]

    if verbose > 0: print(f"Looking for working URL...")
    for url in url_list:
        if verbose > 1: print(f"Trying URL {url}...")
        time.sleep(0.1)
        try:
            res = requests.get(url, stream=True, timeout=1)

        except requests.exceptions.ConnectionError:
            if verbose > 1: print(f"Connection Error")
            continue
        except requests.exceptions.ReadTimeout:
            if verbose > 1: print("Timed out")
            continue
        except OSError:
            if verbose > 1: print(f"OSError")
            continue
        except ValueError:
            if verbose > 1: print(f"Value Error")
            continue

        try:
            if res.ok:
                img_arr = iio.imread(BytesIO(res.content))

                img_arr = tf.image.resize(images=img_arr, size=(image_size, image_size)
                                          , method='bicubic', antialias=True)

                img_arr = np.clip(img_arr / 255, a_min=0.0, a_max=1.0)

                if verbose > 0: print(f"Image retrieved successfully")
                return img_arr
            else:
                continue
        except OSError:
            if verbose > 1: print(f"OSError")
            continue
        except ValueError:
            if verbose > 1: print(f"Value Error")
            continue

    if verbose > 0: print("No image found")
    return np.empty(shape=(1, 1))


def export_dataset_as_json(df, data_type='training'):
    path = EXPORTEDJSON_filepath[data_type]

    print(f"Saving df file as {path}...")
    df.to_json(path_or_buf=path)
    print(f"Saved")


def import_dataset_from_json(data_type='training'):
    path = EXPORTEDJSON_filepath[data_type]

    try:
        print(f"Importing {path}...")
        df = pd.read_json(path)
        print(f"{path} found")

        # df['imageData'] = tf.convert_to_tensor(df['imageData'])
        return df
    except:
        print(f"Error, returning None")
        return None


def import_rawdata(data_type='training', dataset_size=None, verbose=2):
    if dataset_size is None:
        dataset_size = DATASET_numberOfEntries

    path = RAWDATA_filepath[data_type]

    with open(path, 'r') as f:
        data = json.loads(f.read())

    # import the images dataframe and the annotations dataframe, to be merged later

    df_images = pd.json_normalize(data, "images", ["url", "imageId"],
                                  record_prefix='_', errors='ignore')

    if verbose > 1: print("Image IDs retrieved")
    if verbose > 2: print(f"Printing df_images...\n{df_images}")

    df_annotations = pd.json_normalize(data, "annotations", ["labelId", "imageId", "taskId"],
                                       record_prefix='_', errors='ignore')

    if verbose > 1: print("Image annotations retrieved")
    if verbose > 2: print(f"Printing df_annotations...\n{df_annotations}")

    df = pd.merge(df_images, df_annotations, on="_imageId")
    df = df[['_imageId', '_url', '_taskId', '_labelId']]

    # print(type(df.loc[0, '_url']))
    if verbose > 1: print("Image IDs and annotations merged")

    if dataset_size > 0:
        if verbose > 0: print(f"Dataset size reduced to <={dataset_size} (it will be ~{(dataset_size * 0.7):.0f} entries)")
        df = df[0:dataset_size]

    # remove the underscores from the column names
    df.rename(
        columns={
            '_imageId': 'imageId',
            '_url': 'url',
            '_taskId': 'taskId',
            '_labelId': 'labelId'
        }
        , inplace=True)

    df['imageId'] = df['imageId'].astype(dtype=int)
    df['taskId'] = df['taskId'].astype(dtype=int)
    df['labelId'] = df['labelId'].astype(dtype=int)

    # TODO:
    # - create a new variable for the apparel class (each taskid belongs to one and only one apparel class)
    # - download the image from the first URL that's responsive
    # - when the dataset is constructed, save it locally (as json?)

    # apparel class
    apparel_class_map = get_apparel_class_map()
    df['apparel_class'] = df['taskId'].apply(lambda x: apparel_class_map[x])

    # download images
    if verbose > 0: print("Downloading images...")

    if verbose > 1:
        firstStart = time.time()
        I = len(df)

    imagedata_out = list()

    for i, row in df.iterrows():
        imagedata_out.append(get_image_file(row['url']))

        if verbose > 1:
            i_ = i + 1

            end = time.time()
            eta = int(((end - firstStart) / (i_) * (I-i_)))

            if (i_ == 1):
                print("\n")

            endch = ""
            if i_ == I:
                endch = "\n"

            print(f"\r[{i_ / I * 100:.0f}%] ETA: {timeformat(eta)}", end=endch)

    if verbose > 0: print("Images downloaded")
    if verbose > 1: print(f"Took {timeformat(time.time() - firstStart)}")

    #print(type(imagedata_out))
    #print(imagedata_out)

    if verbose > 1: print("Attaching images to df['imageData']...")
    df['imageData'] = imagedata_out
    if verbose > 1: print("Images successfully attached to df['imageData']")

    if verbose > 0: print("Removing data entries with blank images...")
    goodimages_mask = (df['imageData'].apply(lambda x: x.shape) != (1, 1))
    df = df[goodimages_mask]
    if verbose > 0: print(
        f"Data entries with blank images successfully removed (Dataset is now {np.sum(goodimages_mask)} entries)")

    export_dataset_as_json(df, data_type)

    if verbose > 2: print(f"Printing df...\n{df}")
    return df


def get_dataset(data_type='training', verbose = 1):
    dataset = None
    if os.path.exists(EXPORTEDJSON_filepath[data_type]):
        if verbose > 0: print(f"File {EXPORTEDJSON_filepath[data_type]} present, attempting import...")
        dataset = import_dataset_from_json(data_type)
    else:
        if verbose > 0: print(f"File {EXPORTEDJSON_filepath[data_type]} not present, getting raw data...")
        dataset = import_rawdata()

    if dataset is not None:
        if verbose > 0:
            a = dataset['imageData']
            list_to_print = list()
            while True:
                try:
                    list_to_print.append(str(len(a)))
                    a = a[0]
                except:
                    break

            print(f"Images shape is ({', '.join(list_to_print)})")

    return dataset

#import_rawdata()
#import_rawdata('validation')
#Data = get_dataset()

"""
print(f"First row:\n{Data.iloc[0]}")

print(f"Testing import...")
ImportedData = import_dataset_from_json()

print(f"First row:\n{ImportedData.iloc[0]}")

taskMap = get_task_map()
print(f"task map:\n{taskMap}")

apparelClassMap = get_apparel_class_map(taskMap)
print(f"apparel class map: {apparelClassMap}")
"""
