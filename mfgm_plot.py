import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

def fancify_tasks(tasks):
    return [string.capwords(task.split(':')[1].replace("_", " ")) for task in tasks]

def make_training_log_plots(csv_paths, csv_titles, tasks, linestyles=None, alpha_multiplier=None, maxepochs=None, rectify_y_axis=True):
    """

    :param csv_paths: path to csv
    :param csv_titles: good-looking titles for the csvs
    :param tasks: for example ['shoe:up height', 'shoe:age', 'shoe:color']. it will make plots of these tasks (and of the overall loss/accuracy metrics)
    :param linestyles: if specified, allows linestyles to be chosen.
    :param alpha_multiplier: by how much opacity increases for every subsequent dataset. the most opaque dataset will have alpha equal to 1.
    :param maxepochs: set the maximum number of epochs to show.
    :param rectify_y_axis: If true, the y axes' ranges are not matplotlib's defaults.
    :return:
    """

    if linestyles is None:
        l = ((0, (3, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1, 1, 1)))

        linestyles = []
        last_i = len(csv_paths) - 1

        for i in range(len(csv_paths)):
            if i == last_i:
                linestyles.append('solid')
            else:
                linestyles.append(l[i % len(l)])

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

    fancy_tasks = fancify_tasks(tasks)
    
    for i in range(len(tasks)):
        task = tasks[i]
        task_fancy = fancy_tasks[i]

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
            plt.plot(data[train_accuracy_name], label=f"{data_title} Training Accuracy", color=training_color, alpha=alphas[d], linestyle=linestyles[d])
            plt.plot(data[val_accuracy_name], label=f"{data_title} Validation Accuracy", color=validation_color, alpha=alphas[d], linestyle=linestyles[d])

            if last:
                plt.axhline(y=data[randomguess_name][0], linestyle='--', label=f"Random Guess", color=randomguess_color, alpha=alphas[d])

            if most_important:
                if rectify_y_axis:
                    plt.ylim((0.0, 1.0))

                ylocs, ylabels = plt.yticks()
                ylocs = np.append(ylocs, data[randomguess_name][0])
                ylabels = [f"{loc * 100:.0f}%" for loc in ylocs]
                plt.yticks(ylocs, ylabels)

            if last:
                plt.xlabel('epochs')
                plt.xticks(range(0, len(data['epoch']) + 1, 5))

                if maxepochs is not None:
                    plt.xlim((0, maxepochs))

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

            plt.plot(training_loss, label=f"{data_title} Training Loss", color=training_color, alpha=alphas[d], linestyle=linestyles[d])
            plt.plot(validation_loss, label=f"{data_title} Validation Loss", color=validation_color, alpha=alphas[d], linestyle=linestyles[d])

            if most_important and rectify_y_axis:
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
                plt.xticks(range(0, len(data['epoch']) + 1, 5))

                if maxepochs is not None:
                    plt.xlim((0, maxepochs))

                plt.title(f"{versus}: {task_fancy} Loss")
                plt.legend()
                plt.show()
        
    return

"""
def make_conditional_accuracy_bar_graphs(model_path, df, generator, tasks):

    model = keras.models.load_model(model_path)

    if len(df) != generator.batch_size:
        print("len(df) and generator.batch_size are not equal")


    for data_batch, labels_batch in generator:
        predicted_labels = model.predict(data_batch)
        break

    print(predicted_labels)

    #ncols = 2
    #fig, axes = plt.subplots(nrows=np.ceil(len(tasks) / ncols), ncols=ncols)

    #for a, ax in enumerate(axes):
"""

"""
csv_path = 'shoeMultiTaskModel_trainingLog (3).csv'
tasks=('shoe:up height', 'shoe:age', 'shoe:color') 
make_training_log_plots(csv_path, tasks)
"""
