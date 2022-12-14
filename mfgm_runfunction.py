import mfgm_imaterialist as im
import mfgm_multitask as mt
import mfgm_parameters as PARAMS
from datetime import datetime

def run(batch_size = 128,
        epochs = 20,
        private_dense_layers=False,
        reduced_parameters_for_public = False,
        apparel_class = 'all',
        tasks = ('common:apparel_class', 'common:color', 'common:material', 'common:age', 'common:gender'),
        randomize_missing_labels = False,
        purge_bad_images = True,
        micro_dataset = False
        ):

    if purge_bad_images:
        print("Running")
        im.purge_bad_images('training')
        im.purge_bad_images('validation')

    modelname = f"{PARAMS.MFGM_MULTITASK_MODEL_filename}_{datetime.now().strftime('%d-%m-%Y %H-%M-%S')}"
    print(f"Model name: {modelname}")

    dataset_train, number_of_labels, fill_rate_train = mt.get_multitask_subset(
        data_type='training',
        apparel_class=apparel_class,
        tasks=tasks,
        randomize_missing_labels=randomize_missing_labels,
        modelname=modelname,
        micro_dataset=micro_dataset,
        verbose=1)

    dataset_validation, _, fill_rate_val = mt.get_multitask_subset(
        data_type='validation',
        apparel_class=apparel_class,
        tasks=tasks,
        randomize_missing_labels=randomize_missing_labels,
        micro_dataset=micro_dataset,
        verbose=1)

    model, task_reformat = mt.get_untrained_multitask_model(
        tasks_arg=tasks,
        number_of_labels=number_of_labels,
        private_dense_layers=private_dense_layers,
        reduced_parameters_for_public=reduced_parameters_for_public,
        verbose=1)

    # hypothesis: batch_size and epochs need to be VERY high,
    # as most of the data fed is missing labels that cannot train
    mt.train_multitask_model(
        model=model,
        df_train=dataset_train,
        df_validation=dataset_validation,
        tasks=tasks,
        number_of_labels=number_of_labels,
        missing_labels_were_randomized=randomize_missing_labels,
        fill_rate_train=fill_rate_train,
        fill_rate_val=fill_rate_val,
        task_reformat=task_reformat,
        batch_size=batch_size,
        epochs=epochs,
        modelname=modelname,
        verbose=1)


# run()
