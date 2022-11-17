import mfgm_imaterialist as im
import mfgm_multitask as mt


def run():
    im.purge_bad_images('training')
    im.purge_bad_images('validation')

    """
    tasks = ['shoe:gender', 'shoe:type', 'shoe:age', 'shoe:closure type', 'shoe:up height', 'shoe:heel type',
             'shoe:back counter type', 'shoe:material', 'shoe:color', 'shoe:decoration', 'shoe:toe shape', 'shoe:flat type']

    sorted:

    tasks = ['shoe:gender', 'shoe:age', 'shoe:type', 'shoe:up height', 'shoe:back counter type', 'shoe:closure type',
             'shoe:heel type', 'shoe:toe shape', 'shoe:decoration', 'shoe:flat type', 'shoe:color', 'shoe:material']
    """
    # ['shoe:up height', 'shoe:age', 'shoe:color']
    apparel_class = 'all'
    tasks = ['common:apparel_class', 'common:gender', 'common:age', 'common:color', 'common:material']

    randomize_missing_labels = True

    dataset_train, number_of_labels, fill_rate_train = mt.get_multitask_subset(
        data_type='training',
        apparel_class=apparel_class,
        tasks=tasks,
        randomize_missing_labels=randomize_missing_labels,
        verbose=1)

    dataset_validation, _, fill_rate_val = mt.get_multitask_subset(
        data_type='validation',
        apparel_class=apparel_class,
        tasks=tasks,
        randomize_missing_labels=randomize_missing_labels,
        verbose=1)

    model, task_reformat = mt.get_untrained_multitask_model(
        tasks_arg=tasks,
        number_of_labels=number_of_labels,
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
        batch_size=128,
        epochs=60,
        verbose=1)


run()
