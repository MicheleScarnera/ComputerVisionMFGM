import json
import os
import pandas as pd
import computervision_parameters as PARAMS

def timeformat(secs):
    if type(secs) is not int: secs = int(secs)
    # Formats an integer secs into a HH:MM:SS format.
    return f"{str(secs // 3600).zfill(2)}:{str((secs // 60) % 60).zfill(2)}:{str(secs % 60).zfill(2)}"

def dictformat(dict) -> str:
    """

    :param dict:
    :return: Returns a better-looking formatting for a dictionary.
    """

    result = []
    for key, value in dict.items():
        result.append(f"{key} -> {value}")

    result = ",\n".join(result)
    return result

def get_tasks() -> list:
    """

    :return:
    """
    taskMapPath = PARAMS.RAWDATA_taskMapPath

    with open(taskMapPath, 'r') as f:
        taskMap_raw = json.loads(f.read())

    mappings = taskMap_raw['taskInfo']
    result = list()

    for l in mappings:
        result.append(l['taskName'])

    result.sort()

    return result

def get_task_map() -> dict[str, str]:
    """
    :return: Returns a dict[str, str] that maps the taskId (i.e. 12) to the name of the task (i.e. dress:length)
    """
    taskMapPath = PARAMS.RAWDATA_taskMapPath

    with open(taskMapPath, 'r') as f:
        taskMap_raw = json.loads(f.read())

    mappings = taskMap_raw['taskInfo']
    result = dict()

    for l in mappings:
        result[l['taskId']] = l['taskName']

    return result

def get_label_map() -> dict[str, str]:
    """
    :return: Returns a dict[str, str] that maps the labelId (i.e. 3) to the name of the task (i.e. black)
    """
    label_map_path = PARAMS.RAWDATA_labelMapPath

    with open(label_map_path, 'r') as f:
        label_map_path_raw = json.loads(f.read())

    mappings = label_map_path_raw['labelInfo']
    result = dict()

    for l in mappings:
        result[l['labelId']] = l['labelName']

    return result

def get_apparel_class_map(task_map=None) -> dict[str, str]:
    """
    :param task_map: If already computed, you can specify the task map.
    :return: Returns a dict[str, str] that maps the taskId (i.e. 12) to the name of the apparel class (i.e. dress)
    """
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
        result[key] = value.split(":")[0] #string_to_int[value.split(":")[0]]

    return result


def get_apparel_to_all_tasks_map(values_are_names=True) -> dict[str, list]:
    """
    :param values_are_names: If true, the dictionary values will be i.e. ['gender', 'age', 'color', ...]. If false, they will be i.e. ['1', '2', '3', '4', '9', ...]
    :return: Returns a dict[str, list] that maps the name of the apparel class (i.e. shoe) to all of its tasks (i.e. ['gender', 'age', 'color', ...])
    """
    taskMapPath = PARAMS.RAWDATA_taskMapPath

    with open(taskMapPath, 'r') as f:
        taskMap_raw = json.loads(f.read())

    mappings = taskMap_raw['taskInfo']
    result = dict()

    for mapping in mappings:
        spl = mapping['taskName'].split(":")
        key = spl[0]

        if values_are_names:
            value = mapping['taskName']
        else:
            value = mapping['taskId']

        if key not in result:
            result[key] = []

        result[key].append(value)

    return result


def import_dataset_from_json(data_type='training'):
    """
    Copy-and-paste from import_imaterialist cause I don't wanna figure out how to get rid of the circular import (thanks, interpreted language!)
    :param data_type:
    :return:
    """
    path = PARAMS.EXPORTEDJSON_filepath[data_type]

    try:
        return pd.read_json(path)
    except:
        return None

def get_task_to_all_values_map(training_dataset=None, task_map=None, label_map=None) -> dict[int, list]:
    """
    :param training_dataset: If already computed, you can provide the training set.
    :return:Returns a dict[int, list] that maps the task name to all of its possible label values.
    """
    if training_dataset is None:
        training_dataset = import_dataset_from_json()

    if task_map is None:
        task_map = get_task_map()

    if label_map is None:
        label_map = get_label_map()

    result = dict()

    for i, row in training_dataset.iterrows():
        key = task_map[row['taskId']]
        value = label_map[row['labelId']]

        try:
            result[key].add(value)
        except KeyError:
            result[key] = set()
            result[key].add(value)

    return result

"""
print("Apparel Class -> All its tasks:")
print(dictformat(get_apparel_to_all_tasks_map()))
print(" ")
print("Task ID -> All its labels:")
print(dictformat(get_task_to_all_values_map()))
"""