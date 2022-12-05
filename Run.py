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

"""
# Training the model
"""

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

"""
# Plots
After training a model, all of its output will be saved in a folder inside `trainedModels`, with the format `multiTaskModel_[date] [time]`. The contents of that folder can be used to make plots.
"""

import mfgm_plot as plot

"""
# Training logs
The following parameters are used:
- `log_tasks`: Which task(s) you want to see plots of.
- `log_paths`: The path(s) of the training logs. Refers to the folder named `multiTaskModel_[date] [time]` that is _inside_ `multiTaskModel_[date] [time]`.
- `log_titles`: The title(s) of the training logs. Must be the same length as `log_paths`.
- `maxepochs`: How many epochs should the plot show.
"""

log_tasks = ['common:apparel_class']
log_paths = ['finalModels/gender/trainingLog.csv',
            'finalModels/age/trainingLog.csv',
            'finalModels/material/trainingLog.csv',
            'finalModels/color/trainingLog.csv',
            'finalModels/apparel/trainingLog.csv']
log_titles = ['Apparel+Gender', 'Apparel+Age', 'Apparel+Material', 'Apparel+Color', 'Apparel']
maxepochs = 20

plot.make_training_log_plots(csv_paths=log_paths,
                             csv_titles=log_titles,
                             tasks=log_tasks,
                             maxepochs=maxepochs)

"""
# Confusion Matrix
The following parameters are used:
- `model_path`: The path of the model.
- `tasktolabels_path`: Path of the `tasktolabels.json` file that is in the upper `multiTaskModel_[date] [time]` folder.
- `task`: Which task to do the confusion matrix on.
"""

model_path = 'finalModels/apparel/apparel'
tasktolabels_path = 'finalModels/apparel/tasktolabelsmap.json'
task = 'common:apparel_class'

plot.make_confusion_matrix(model_path=model_path,
                           tasktolabels_path=tasktolabels_path,
                           task=task)