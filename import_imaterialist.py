import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as plt
import imageio.v3 as iio
import requests
import json
from io import BytesIO
import time

def get_task_map() -> dict[int, str]:
    taskMapPath = "iMaterialist/fgvc4_iMat.task_map.json"

    with open(taskMapPath, 'r') as f:
        taskMap_raw = json.loads(f.read())

    mappings = taskMap_raw['taskInfo']
    result = dict()

    for l in mappings:
        result[int(l['taskId'])] = l['taskName']

    return result


def get_apparel_class_map(task_map = None) -> dict[int, int]:
    if (task_map == None):
        task_map = get_task_map()

    string_to_int = {
        'outerwear': 1,
        'dress': 2,
        'pants': 3,
        'shoe': 4
    }

    result = dict()
    for key, value in task_map.items():
        result[key] = string_to_int[value.split(":")[0]]

    return result


def get_image_file(url_list, image_size = 256):
    if type(url_list) is str:
        url_list = [url_list]

    print(f"Looking for working URL...")
    for url in url_list:
        print(f"Trying URL {url}...")
        time.sleep(0.1)
        try:
            res = requests.get(url, stream=True, timeout=5)

        except requests.exceptions.ConnectionError:
            print(f"Connection Error")
            continue
        except requests.exceptions.ReadTimeout:
            print("Timed out")
            continue
        except OSError:
            print(f"OSError")
            continue
        except ValueError:
            print(f"Value Error")
            continue

        try:
            if res.ok:
                img_arr = iio.imread(BytesIO(res.content))

                img_arr = tf.image.resize(images=img_arr, size=(image_size, image_size)
                                          , method='bicubic', antialias=True)

                img_arr = img_arr / 255

                print(f"Image retrieved successfully")
                return img_arr
            else:
                continue
        except OSError:
            print(f"OSError")
            continue
        except ValueError:
            print(f"Value Error")
            continue

    print("No image found")
    return np.empty(shape=(1, 1))

def import_data(small_dataset = True, verbose = False):
    trainingPath = "iMaterialist/fgvc4_iMat.train.data.json"

    with open(trainingPath, 'r') as f:
        data = json.loads(f.read())

    # import the images dataframe and the annotations dataframe, to be merged later

    df_images = pd.json_normalize(data, "images", ["url", "imageId"],
                                  record_prefix='_', errors='ignore')

    print("Image IDs retrieved")
    if verbose: print(f"Printing df_images...\n{df_images}")

    df_annotations = pd.json_normalize(data, "annotations", ["labelId", "imageId", "taskId"],
                                       record_prefix='_', errors='ignore')

    print("Image annotations retrieved")
    if verbose: print(f"Printing df_annotations...\n{df_annotations}")

    df = pd.merge(df_images, df_annotations, on="_imageId")
    df = df[['_imageId', '_url', '_taskId', '_labelId']]

    #print(type(df.loc[0, '_url']))
    print("Image IDs and annotations merged")

    if small_dataset:
        l = 5
        print(f"Dataset size reduced to {l}")
        df = df[0:l]

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

    # apparel class
    apparel_class_map = get_apparel_class_map()
    df['apparel_class'] = df['taskId'].apply(lambda x: apparel_class_map[x])

    # download images
    print("Downloading images...")
    imagedata_out = list()
    for index, row in df.iterrows():
        imagedata_out.append(get_image_file(row['url']))

    print("Images downloaded")
    print(type(imagedata_out))
    print(imagedata_out)

    if verbose: print(f"Printing df...\n{df}")
    return df

Data = import_data()

print(f"First row:\n{Data.iloc[0]}")

"""
taskMap = get_task_map()
print(f"task map:\n{taskMap}")

apparelClassMap = get_apparel_class_map(taskMap)
print(f"apparel class map: {apparelClassMap}")
"""