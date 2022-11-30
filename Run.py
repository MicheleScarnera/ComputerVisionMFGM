from mfgm_runfunction import run

"""
# Hyperparameters (that were actively examined)
- `lighting_round`: Used for testing purposes. Forces epochs to be 2, decimates the sample size and skips data integrity checks.
- `batch_size`: Batch size of training. Set to 128 as per standard.
- `epochs`: Number of epochs. Set to 20 due to limted scope.
- `private_dense_layers`: If True, the multi-task learner has dense layers that are inside each task, and thus don't interact with other tasks. If False, the dense layers are instead placed in a part of the network that all tasks use.
- `reduced_parameters_for_public`: If True, the public dense layers are less wide, resulting in ~1.5m parameters, similarly to what a private-dense-layers model would have. If False, the model will have ~3mln parameters.
- `apparel_class`: Which apparel classes to consider.
- `tasks`: Which tasks to consider. All of tasks are `['common:apparel_class', 'common:color', 'common:material', 'common:age', 'common:gender']`.
- `randomize_missing_labels`: Most images don't have labels for most tasks. If `randomize_missing_labels` is True, they are assigned a random value. If False, they are assigned `-1`.
"""

lighting_round = False

batch_size = 128
epochs = 20
private_dense_layers = False
reduced_parameters_for_public = False
apparel_class = 'all'
tasks = ['common:apparel_class']
randomize_missing_labels = False

if lighting_round:
    epochs = 2
    purge_bad_images = False
    micro_dataset = True
else:
    purge_bad_images = True
    micro_dataset = False

run(batch_size = batch_size,
    epochs = epochs,
    private_dense_layers = private_dense_layers,
    reduced_parameters_for_public = reduced_parameters_for_public,
    apparel_class = apparel_class,
    tasks = tasks,
    randomize_missing_labels = randomize_missing_labels,
    purge_bad_images = purge_bad_images,
    micro_dataset = micro_dataset
    )
