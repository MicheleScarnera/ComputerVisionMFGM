import string

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

def make_training_log_plots(csv_paths, csv_titles, tasks, alpha_multiplier=None):
    """

    :param csv_paths: path to csv
    :param csv_titles: good-looking titles for the csvs
    :param tasks: for example ['shoe:up height', 'shoe:age', 'shoe:color']. it will make plots of these tasks (and of the overall loss/accuracy metrics)
    :param alpha_multiplier: by how much opacity increases for every subsequent dataset. the most opaque dataset will have alpha equal to 1.
    :return:
    """

    if alpha_multiplier is None:
        alpha_multiplier = (1. - 1. / len(csv_paths)) ** (-2.)

    datas = []

    for path in csv_paths:
        datas.append(pd.read_csv(path))

    alphas = np.array([alpha_multiplier**n for n in range(len(csv_paths))])
    alphas = alphas / np.max(alphas)

    versus = " vs ".join(csv_titles)

    training_color = 'blue'
    validation_color = 'orange'
    randomguess_color = 'green'
    
    for i in range(len(tasks)):
        task = tasks[i]
        task_fancy = string.capwords(task.split(':')[1].replace("_", " "))

        train_accuracy_name = task + '_accuracy'
        val_accuracy_name = 'val_' + task + '_accuracy'
        randomguess_name = task + '_randomguess'

        train_loss_name = task + '_loss'
        val_loss_name = 'val_' + task + '_loss'

        # Plot the accuracies
        for d, data in enumerate(datas):
            most_important = alphas[d] == 1.0
            last = (d + 1) == len(datas)

            # define data_title
            data_title = csv_titles[d]

            # Plot the accuracy
            plt.plot(data[train_accuracy_name], label=f"{data_title} Training Accuracy", color=training_color, alpha=alphas[d])
            plt.plot(data[val_accuracy_name], label=f"{data_title} Validation Accuracy", color=validation_color, alpha=alphas[d])

            if last:
                plt.plot(data[randomguess_name], linestyle='--', label=f"Random Guess", color=randomguess_color, alpha=alphas[d])

            if most_important:
                plt.ylim((0.0, 1.0))
                ylocs, ylabels = plt.yticks()
                ylocs = np.append(ylocs, data[randomguess_name][0])
                ylabels = [f"{loc * 100:.0f}%" for loc in ylocs]
                plt.yticks(ylocs, ylabels)

            if last:
                plt.xlabel('epochs')
                plt.xticks(range(0, len(data['epoch']), 5))

                plt.title(f"{versus}: {task_fancy} Accuracy")
                plt.legend()
                plt.show()
        
        # Plot the losses
        for d, data in enumerate(datas):
            most_important = alphas[d] == 1.0
            last = (d + 1) == len(datas)

            # define data_title
            data_title = csv_titles[d]

            training_loss = data[train_loss_name]
            training_loss = training_loss[np.isfinite(training_loss)]

            validation_loss = data[val_loss_name]
            validation_loss = validation_loss[np.isfinite(validation_loss)]

            plt.plot(training_loss, label=f"{data_title} Training Loss", color=training_color, alpha=alphas[d])
            plt.plot(validation_loss, label=f"{data_title} Validation Loss", color=validation_color, alpha=alphas[d])

            if most_important:
                ceiling = np.max(training_loss) * 1.5
                step = np.min(
                    [np.round(0.5 * (np.max(training_loss) - np.min(training_loss)), decimals=1)
                        , 0.5]
                )
                plt.ylim((0.0, ceiling))
                ylocs, ylabels = plt.yticks()
                ylocs = np.arange(start=0.0, stop=ceiling, step=step)
                ylabels = [f"{loc:.2f}" for loc in ylocs]
                plt.yticks(ylocs, ylabels)

            if last:
                plt.xlabel('epochs')
                plt.xticks(range(0, len(data['epoch']), 5))

                plt.title(f"{versus}: {task_fancy} Loss")
                plt.legend()
                plt.show()
        
    return

"""
csv_path = 'shoeMultiTaskModel_trainingLog (3).csv'
tasks=('shoe:up height', 'shoe:age', 'shoe:color') 
make_training_log_plots(csv_path, tasks)
"""
