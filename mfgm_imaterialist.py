import mfgm_parameters as PARAMS
import mfgm_misc as MISC

import os
import os.path
import io
import PIL.Image as pil_image
import numpy as np
import pandas as pd
import requests
import json
import time

import warnings


def save_image_file(path, url_list, verbose = 0):
    """
    Attempt to download an image, from a list of URLs, to a path.

    :param path: Path to save the image to.
    :param url_list: List of URL mirrors. Can also be a single URL.
    :param verbose: Level of verbosity
    :return:
    """
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


def purge_bad_images(data_type='training'):
    """
    Deletes every image that cannot be loaded by PIL.Image

    :param data_type: Images to look at
    :return: None
    """
    warnings.filterwarnings("error")

    path = PARAMS.IMAGES_filepath[data_type]

    print_first_part = f"Purging bad images from {path}..."
    print(print_first_part, end="\r")
    for (root, _, filenames) in os.walk(path):
        before = len(filenames)
        removed = 0
        for i, file_name in enumerate(filenames):
            delete_file = False

            file_path = os.path.join(root, file_name)
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    try:
                        img = pil_image.open(io.BytesIO(f.read())).load()
                    except KeyboardInterrupt as interrupt:
                        raise KeyboardInterrupt()
                    except:
                        delete_file = True

            if delete_file:
                os.remove(file_path)
                removed = removed + 1

            final_step = (i+1) == before

            if i % 10 == 0 or final_step:
                print("                               ", end="\r")
                endch = "\r"
                if (i+1) == before:
                    endch = "\n"

                print(f"{print_first_part} {(i+1) / before * 100:.1f}%, found {removed}", end=endch)

        break

    print(f"{removed} images removed from {before} images")
    warnings.filterwarnings("default")


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

        df['imageId'] = df['imageId'].astype(dtype=str)
        df['taskId'] = df['taskId'].astype(dtype=str)
        df['labelId'] = df['labelId'].astype(dtype=str)
        return df
    except:
        print(f"Error, returning None")
        return None


def kill_rare_labels(data, task_name, threshold, label_map=None, precalculated_bad_labels=None, verbose=1):
    """
    Originally written by Michelle Luijten

    Given a task name and a percentage threshold, removes every label of that task that is present less often than the threshold, and groups them into the "other" label.

    :param data: DataFrame
    :param task_name: Name of the task
    :param threshold: Threshold under which uncommon labels are grouped into "other"
    :param label_map: If already computed, you can provide the label id to label name map.
    :param precalculated_bad_labels: In the case of validation sets, one must already have computed the bad labels on the training set, and must provided here
    :param verbose: Level of verbosity
    :return: (DataFrame, list) tuple with the new dataset and the label IDs that were removed
    """
    if label_map is None:
        label_map = MISC.get_label_map()

    # select subset of data only containing these task-ids
    subset = data[data['taskName'] == task_name]

    # count how often each label id occurs
    a = subset.groupby('labelId').count()

    # calculate ratio
    df = pd.DataFrame({'labelId': a.index, 'counts': a['imageId']}).reset_index(drop=True)
    df['ratio'] = df['counts'] / sum(df['counts'])

    # group every label that is used in less than 1% (threshold) of the cases
    if precalculated_bad_labels is None:
        to_replace = list(df.loc[df['ratio'] < threshold, 'labelId'])
    else:
        to_replace = list(df.loc[df['labelId'].isin(precalculated_bad_labels), 'labelId'])

    l = len(to_replace)
    if l > 0 and verbose > 0:
        bad_labels = [label_map[x] for x in to_replace]

        print(f"The task {task_name} has {l} labels (~{l / len(df) * 100:.0f}% of labels) that are present less than {threshold * 100:.0f}% of the time: {bad_labels}")

    # set this labelId to 999
    data.loc[data['labelId'].isin(to_replace), 'labelId'] = "999"

    data['labelName'] = data['labelId'].apply(lambda x: label_map[x])

    return data, to_replace

def import_rawdata(
        data_type='training',
        dataset_size=None,
        freeloader_mode=True,
        attempt_downloading_images=True,
        delete_orphan_entries=True,
        add_common_tasks=True,
        kill_common_task_label_frequency_threshold=0.01,
        precalculated_bad_labels=None,
        add_apparel_class_as_task=True,
        save_json=True,
        verbose=2):
    """
    Forcibly creates a new [data_type]Data.json file. Returns the created dataframe.

    :param data_type: Either 'training' or 'validation'. 'test' is currently not supported
    :param dataset_size: How many entries to look into. If not specified, DATASET_numberOfEntries is used
    :param freeloader_mode: If enough images are already present, no downloading is attempted and the json is constructed around the available files
    :param attempt_downloading_images: If false, images are not downloaded no matter what
    :param delete_orphan_entries: If true, entries that don't have their image downloaded are deleted
    :param add_common_tasks: If true, any entry that specifies common tasks like 'shoe:color' is duplicated, with its task name changed to 'color'
    :param kill_common_task_label_frequency_threshold: Among common tasks, if a label is present less than X% of the time, it is replaced with 'other'
    :param precalculated_bad_labels: Used by non-training sets. Used to make sure other sets remove the same labels as the training set.
    :param add_apparel_class_as_task: If true, the apparel class is added as a task
    :param save_json: If true, the [data_type]Data.json file is saved.
    :param verbose: Level of verbosity.
    :return: (DataFrame, list) tuple with the new dataset and the label IDs that were removed
    """
    folder_path = PARAMS.IMAGES_filepath[data_type]
    if verbose > 0: print(f"### IMPORTING DATA INTO {folder_path} ###\n")

    if freeloader_mode:
        if verbose > 0: print("""Checking if freeloading is possible...""")

        f = []
        for (_, _, filenames) in os.walk(folder_path):
            f.extend(filenames)
            break

        n = 20
        if len(f) < n:
            if verbose > 0: print(f"Less than {n} files found in {folder_path}, FREELOADING has been DISABLED\n")
            freeloader_mode = False
        else:
            if verbose > 0: print(f"""{n} or more files found in {folder_path}, FREELOADING has been kept ENABLED
NOTE: If you want to download more images than you have locally, delete the present image files or set freeloader_mode to False
""")

    if dataset_size is None and delete_orphan_entries:
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

    if freeloader_mode:
        if verbose > 0: print(f"Freeloader mode is enabled, dataset size will be of however many images are found")
    else:
        if (dataset_size is not None) and (dataset_size > 0):
            if verbose > 1: print("Freeloader mode is disabled, hence images that weren't found will be downloaded")
            if verbose > 0: print(f"Dataset will have ~{(dataset_size * 0.7):.0f} entries")
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

    df['imageId'] = df['imageId'].astype(dtype=str)
    df['taskId'] = df['taskId'].astype(dtype=str)
    df['labelId'] = df['labelId'].astype(dtype=str)

    # apparel class
    apparel_class_map = MISC.get_apparel_class_map()
    df['apparelClass'] = df['taskId'].apply(lambda x: apparel_class_map[x])
    
    # task name
    task_map = MISC.get_task_map()
    df['taskName'] = df['taskId'].apply(lambda x: task_map[x])

    # label name
    label_map = MISC.get_label_map()
    df['labelName'] = df['labelId'].apply(lambda x: label_map[x])

    try:
        os.mkdir(folder_path)
    except FileExistsError as error:
        if verbose > 1: print(f"{folder_path} folder already exists")
    else:
        if verbose > 1: print(f"{folder_path} folder created")

    if (not freeloader_mode) and verbose > 0: print("Downloading images...")

    if verbose > 1:
        firstStart = time.time()
        I = len(df)
        expected = np.zeros(shape=I)

    image_path_func = lambda x: f"{folder_path}/{x}.jpg"

    df['imageName'] = df['imageId'].apply(lambda x: f"{x}.jpg")

    # which image files are present?
    # (i tried doing this with pandas.apply() instead of this "dumber" approach,
    # but i don't think os.path.isfile() really works in there)
    possible_image_ids = np.unique(df['imageId'])
    was_found = dict()
    for id in possible_image_ids:
        was_found[id] = os.path.isfile(image_path_func(id))

    image_found = df['imageId'].apply(lambda x: was_found.get(x, False))

    if (not freeloader_mode) and verbose > 1: print(f"Before downloading any new images, {np.sum(image_found)} images were present")

    if (not freeloader_mode) and attempt_downloading_images:
        # iterating over df and not df_images is inefficient. however, df_images does not have the apparelClass column.
        for i, row in df[~image_found].iterrows():
            image_path = image_path_func(row['imageId'])  # {row['apparelClass']}/

            image_found[i] = save_image_file(image_path, row['url'])

            if verbose > 1:
                i_ = i + 1

                end = time.time()
                average_time = (end - firstStart) / i_
                eta = int((average_time * (I - i_)))
                expected[i] = average_time * I

                endch = ""
                if i_ == I:
                    endch = "\n"

                print("\r                                                                                         ",
                      end="")
                print(f"\r[{i_ / I * 100:.0f}%] ETA: {MISC.timeformat(eta)}", end=endch)

    else:
        if verbose > 0:
            print(f"Due to freeloader mode being enabled or attempt_downloading_images being false, no new image was downloaded")

    if (not freeloader_mode) and verbose > 1:
        a = np.mean(expected)
        b = 2 * np.std(expected)
        print(f"Took {MISC.timeformat(time.time() - firstStart)}. Expectation was ({MISC.timeformat(a - b)}, {MISC.timeformat(a + b)})")

    if (not freeloader_mode) and verbose > 0: print("Images downloaded successfully")

    if delete_orphan_entries:
        if verbose > 0: print("Removing data entries with no images...")
        df = df[image_found]
        if verbose > 0: print(
            f"Data entries with blank images successfully removed. Dataset (with images) is now {np.sum(image_found)} entries")
    else:
        if verbose > 0: print(f"Dataset (with images) is now {np.sum(image_found)} entries")


    # keep only the columns we care about. the 'url' column is a waste of space
    df.drop(columns=['url'], inplace=True)

    if add_common_tasks or add_apparel_class_as_task:
        task_map_reverse = MISC.get_task_map(reverse=True)
        label_map_reverse = MISC.get_label_map(reverse=True)

    # add common tasks
    if add_common_tasks:
        if verbose > 0: print(f"Adding rows for common tasks...")

        common_mapper = MISC.get_mapper_to_common_task()

        df_with_common_tasks = df.copy()
        df_with_common_tasks['taskName'] = df_with_common_tasks['taskName'].apply(lambda x: common_mapper[x])
        df_with_common_tasks = df_with_common_tasks[df_with_common_tasks['taskName'].apply(lambda x: x is not None)]

        # apply correct task/label id
        df_with_common_tasks['taskId'] = df_with_common_tasks['taskName'].apply(lambda x: task_map_reverse[x])
        df_with_common_tasks['labelId'] = df_with_common_tasks['labelName'].apply(lambda x: label_map_reverse[x])

        df = pd.concat(objs=(df, df_with_common_tasks), ignore_index=True)
        df_with_common_tasks = None

        common_tasks = np.unique([value for value in common_mapper.values() if value is not None])

        if data_type != 'training' and precalculated_bad_labels is None:
            if verbose > 0:
                print(f"""
WARNING: A {data_type} set was not given the precalculated_bad_labels argument.
It will be calculated to make sure the removed labels match with the training set.
You probably already have calculated the training test somewhere before.
""")

            _, precalculated_bad_labels = import_rawdata('training',
                                                         dataset_size=-1,
                                                         freeloader_mode=True,
                                                         attempt_downloading_images=False,
                                                         save_json=False,
                                                         verbose=0)
        elif data_type == 'training':
            precalculated_bad_labels = None

        if data_type == 'training':
            p = None
        else:
            p = precalculated_bad_labels

        for common_task in common_tasks:
            df, b = kill_rare_labels(data=df,
                                    task_name=common_task,
                                    threshold=kill_common_task_label_frequency_threshold,
                                    precalculated_bad_labels=p,
                                    label_map=label_map)

            if precalculated_bad_labels is not None:
                for _b in b:
                    precalculated_bad_labels.append(_b)

    if add_apparel_class_as_task:
        if verbose > 0: print(f"Adding apparel_class task...")

        df_first_time_image_id = df[~df.duplicated(subset='imageId', keep='first')].copy()

        df_first_time_image_id['taskName'] = 'common:apparel_class'
        df_first_time_image_id['labelName'] = df_first_time_image_id['apparelClass']

        # apply correct task/label id
        df_first_time_image_id['taskId'] = df_first_time_image_id['taskName'].apply(lambda x: task_map_reverse[x])
        df_first_time_image_id['labelId'] = df_first_time_image_id['labelName'].apply(lambda x: label_map_reverse[x])

        df = pd.concat(objs=(df, df_first_time_image_id), ignore_index=True)
        df_first_time_image_id = None

    if save_json:
        export_dataset_as_json(df, data_type)

    if verbose > 2: print(f"Printing df...\n{df}")

    return df, precalculated_bad_labels


def get_dataset(data_type='training', verbose = 1) -> pd.DataFrame:
    """
    Returns a pandas.DataFrame containing the non-image data.
    This dataframe is stored in the [...]Data.json file.
    If the file is not found, images are downloaded and the file is created.

    :param data_type: Either 'training' or 'validation'
    :param verbose: Level of verbosity.
    :return: a pandas.DataFrame
    """

    dataset = None
    if os.path.exists(PARAMS.EXPORTEDJSON_filepath[data_type]):
        if verbose > 0: print(f"File {PARAMS.EXPORTEDJSON_filepath[data_type]} present, attempting import...")
        dataset = import_dataset_from_json(data_type)
    else:
        if verbose > 0: print(f"File {PARAMS.EXPORTEDJSON_filepath[data_type]} not present, getting raw data...")
        dataset, _ = import_rawdata(data_type)

    return dataset

