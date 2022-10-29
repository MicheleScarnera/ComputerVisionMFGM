import pandas as pd
import json


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


def import_data():
    trainingPath = "iMaterialist/fgvc4_iMat.train.data.json"

    with open(trainingPath, 'r') as f:
        data = json.loads(f.read())

    # import the images dataframe and the annotations dataframe, to be merged later

    df_images = pd.json_normalize(data, "images", ["url", "imageId"],
                                  record_prefix='_', errors='ignore')

    print(f"Printing df_images...\n{df_images}")

    df_annotations = pd.json_normalize(data, "annotations", ["labelId", "imageId", "taskId"],
                                       record_prefix='_', errors='ignore')

    print(f"Printing df_annotations...\n{df_annotations}")

    df = pd.merge(df_images, df_annotations, on="_imageId")
    df = df[['_imageId', '_url', '_taskId', '_labelId']]

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

    apparel_class_map = get_apparel_class_map()
    df['apparel_class'] = df['taskId'].apply(lambda x: apparel_class_map[x])

    print(f"Printing df...\n{df}")
    return df



Data = import_data()

print(f"First row:\n{Data.iloc[0]}")

"""
taskMap = get_task_map()
print(f"task map:\n{taskMap}")

apparelClassMap = get_apparel_class_map(taskMap)
print(f"apparel class map: {apparelClassMap}")
"""