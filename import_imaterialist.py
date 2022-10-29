import pandas as pd
import json

def ImportData():
    trainingPath = "iMaterialist/fgvc4_iMat.train.data.json"

    with open(trainingPath, 'r') as f:
        data = json.loads(f.read())

    df_images = pd.json_normalize(data, "images", ["url", "imageId"],
                                  record_prefix='_', errors='ignore')

    print(f"Printing df_images...\n{df_images}")

    df_annotations = pd.json_normalize(data, "annotations", ["labelId", "imageId", "taskId"],
                                       record_prefix='_', errors='ignore')

    print(f"Printing df_annotations...\n{df_annotations}")

    df = pd.merge(df_images, df_annotations, on="_imageId")
    df = df[['_imageId', '_url', '_taskId', '_labelId']]
    """
    for i, column_name in enumerate(df.columns):
        df.columns[i] = column_name.replace("_", "")
    """

    df.rename(
        {
            '_imageId': 'imageId',
            '_url': 'url',
            '_taskId': 'taskId',
            '_labelId': 'labelId'
        }
        , inplace=True)

    print(f"Printing df...\n{df}")
    return df

Data = ImportData()

print(f"First row:\n{Data.iloc[0]}")