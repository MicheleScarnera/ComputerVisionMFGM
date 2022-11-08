import computervision_parameters as PARAMS
import computervision_misc as MISC

import os.path
import numpy as np
import pandas as pd
import requests
import json
import time


def save_image_file(path,url_list, verbose = 0):
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
            if res.ok and '<' not in res.text[0:20] and len(res.content) > 5000:
                with open(path, 'wb') as f:
                    f.write(res.content)

                return True
            else:
                continue
        except OSError:
            if verbose > 1: print(f"OSError")
            continue
        except ValueError:
            if verbose > 1: print(f"Value Error")
            continue

    if verbose > 0: print("No image found")
    return False #np.empty(shape=(1, 1))


def export_dataset_as_json(df, data_type='training'):
    path = PARAMS.EXPORTEDJSON_filepath[data_type]

    print(f"Saving df file as {path}...")
    df.to_json(path_or_buf=path)
    df.to_csv(path_or_buf=f"{data_type}Data.csv")
    print(f"Saved")


def import_dataset_from_json(data_type='training'):
    path = PARAMS.EXPORTEDJSON_filepath[data_type]

    try:
        print(f"Importing {path}...")
        df = pd.read_json(path)
        print(f"{path} found")

        # df['imageData'] = tf.convert_to_tensor(df['imageData'])
        return df
    except:
        print(f"Error, returning None")
        return None


def import_rawdata(data_type='training', dataset_size=None, verbose=2) -> pd.DataFrame:
    """
    Forcibly creates a new [data_type]Data.json file. Returns the created dataframe.

    :param data_type: Either 'training' or 'validation'. 'test' is currently not supported
    :param dataset_size: How many entries to look into. If not specified, DATASET_numberOfEntries is used
    :param verbose: Level of verbosity.
    :return: a pandas.DataFrame
    """
    if dataset_size is None:
        dataset_size = PARAMS.DATASET_numberOfEntries

    path = PARAMS.RAWDATA_filepath[data_type]

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

    if verbose > 1: print("Image IDs and annotations merged")

    if dataset_size > 0:
        if verbose > 0: print(f"Dataset size reduced to <={dataset_size} (it will be ~{(dataset_size * 0.7):.0f} entries)")
        df = df[0:dataset_size]

    # remove the underscores from the column names
    nomenclature = {
            '_imageId': 'imageId',
            '_url': 'url',
            '_taskId': 'taskId',
            '_labelId': 'labelId'
        }

    df.rename(columns=nomenclature, inplace=True)
    df_images.rename(columns=nomenclature, inplace=True)

    df['imageId'] = df['imageId'].astype(dtype=int)
    df['taskId'] = df['taskId'].astype(dtype=int)
    df['labelId'] = df['labelId'].astype(dtype=int)

    # apparel class
    apparel_class_map = MISC.get_apparel_class_map()
    df['apparel_class'] = df['taskId'].apply(lambda x: apparel_class_map[x])

    # download images
    folder_path = PARAMS.IMAGES_filepath[data_type]

    try:
        os.mkdir(folder_path)
    except FileExistsError as error:
        if verbose > 1: print(f"{folder_path} folder already exists")
    else:
        if verbose > 1: print(f"{folder_path} folder created")

    for key in np.unique(list(apparel_class_map.values())):
        try:
            other_path = f"{folder_path}/{key}"
            os.mkdir(other_path)
        except FileExistsError as error:
            if verbose > 1: print(f"{other_path} folder already exists")
        else:
            if verbose > 1: print(f"{other_path} folder created")

    if verbose > 0: print("Downloading images...")

    if verbose > 1:
        firstStart = time.time()
        I = len(df)
        expected = np.zeros(shape=I)

    image_found = np.empty(shape = len(df), dtype = bool)

    #TODO: iterating over df and not df_images is inefficient. however, df_images does not have the apparel_class column.
    for i, row in df.iterrows():
        image_path = f"{PARAMS.IMAGES_filepath[data_type]}/{row['apparel_class']}/{row['imageId']}.jpg"

        if not os.path.isfile(image_path):
            image_found[i] = save_image_file(image_path, row['url'])
        else:
            image_found[i] = True

        if verbose > 1:
            i_ = i + 1

            end = time.time()
            average_time = (end - firstStart) / i_
            eta = int((average_time * (I-i_)))
            expected[i] = average_time * I

            endch = ""
            if i_ == I:
                endch = "\n"

            print("\r                                                                                         ", end="")
            print(f"\r[{i_ / I * 100:.0f}%] ETA: {MISC.timeformat(eta)}", end=endch)

    if verbose > 1:
        a = np.mean(expected)
        b = 2 * np.std(expected)
        print(f"Took {MISC.timeformat(time.time() - firstStart)}. Expectation was ({MISC.timeformat(a - b)}, {MISC.timeformat(a + b)})")

    if verbose > 0: print("Images downloaded successfully")

    df['imageWasFound'] = image_found

    if verbose > 0: print("Removing data entries with no images...")
    df = df[image_found]
    if verbose > 0: print(
        f"Data entries with blank images successfully removed (Dataset is now {np.sum(image_found)} entries)")

    #keep only the columns we care about. 'url' and 'imageWasFound' are a waste of space
    df.drop(columns=['url', 'imageWasFound'], inplace=True)

    export_dataset_as_json(df, data_type)

    if verbose > 2: print(f"Printing df...\n{df}")
    return df


def get_dataset(data_type='training', verbose = 1) -> pd.DataFrame:
    """
    Returns a pandas.DataFrame containing the non-image data.
    This dataframe is stored in the [...]Data.json file.
    If the file is not found, images are downloaded and the file is created.

    :param data_type: Either 'training' or 'validation'. 'test' is currently not supported
    :param verbose: Level of verbosity.
    :return: a pandas.DataFrame
    """

    dataset = None
    if os.path.exists(PARAMS.EXPORTEDJSON_filepath[data_type]):
        if verbose > 0: print(f"File {PARAMS.EXPORTEDJSON_filepath[data_type]} present, attempting import...")
        dataset = import_dataset_from_json(data_type)
    else:
        if verbose > 0: print(f"File {PARAMS.EXPORTEDJSON_filepath[data_type]} not present, getting raw data...")
        dataset = import_rawdata(data_type)

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
