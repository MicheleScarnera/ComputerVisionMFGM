import mfgm_imaterialist as im
import mfgm_multitask as mt


def run():
    purge = True

    apparel_class = 'all'
    tasks = ['common:apparel_class', 'common:color', 'common:material', 'common:age', 'common:gender']
    batch_size = 128
    epochs = 60
    randomize_missing_labels = False

    private_dense_layers = False

    if purge:
        print("Running")
        im.purge_bad_images('training')
        im.purge_bad_images('validation')

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
        private_dense_layers=private_dense_layers,
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
        verbose=1)


run()
