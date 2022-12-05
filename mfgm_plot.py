import string
import json

import numpy as np
import pandas as pd
import seaborn as sns
import keras

import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors

from sklearn.metrics import ConfusionMatrixDisplay as confusion_matrix
from sklearn.linear_model import LinearRegression

import mfgm_multitask as multitask
sns.set_theme()


def fancify_tasks(tasks):
    return [string.capwords(task.split(':')[1].replace("_", " ")) for task in tasks]


def make_training_log_plots(csv_paths, csv_titles, tasks, alpha_multiplier=None, training_alpha_multiplier = 0.5, maxepochs=None, rectify_y_axis=False):
    """

    :param csv_paths: paths to csv
    :param csv_titles: good-looking titles for the csvs
    :param tasks: for example ['shoe:up height', 'shoe:age', 'shoe:color']. it will make plots of these tasks (and of the overall loss/accuracy metrics)
    :param alpha_multiplier: by how much opacity increases for every subsequent dataset. the most opaque dataset will have alpha equal to 1.
    :param training_alpha_multiplier: multiplier for the training loss/accuracy opacity.
    :param maxepochs: set the maximum number of epochs to show.
    :param rectify_y_axis: If true, the y axes' ranges are not matplotlib's defaults.
    :return:
    """

    if len(csv_titles) < len(csv_paths):
        print("There are not enough titles for all the paths that have been given. csv_titles will be padded with '???'s.")

        while len(csv_titles) < len(csv_paths):
            csv_titles.append("???")

    if alpha_multiplier is None:
        alpha_multiplier = 1. #(1. - 1. / len(csv_paths)) ** (-2.)

    datas = []

    for path in csv_paths:
        datas.append(pd.read_csv(path))

    alphas = np.array([alpha_multiplier**n for n in range(len(csv_paths))])
    alphas = alphas / np.max(alphas)

    versus = " vs ".join(csv_titles)

    # figure out what colors to use
    hue_start = 0.33
    hue_increment = 1. / (len(csv_paths) + 1)
    saturation = 0.8
    value = 0.8

    colors = []
    hue_current = hue_start

    for d in range(len(csv_paths)):
        new_color = plt_colors.hsv_to_rgb((hue_current, saturation, value))
        colors.append(new_color)

        hue_current = hue_current + hue_increment
        hue_current = hue_current - np.floor(hue_current)

    randomguess_color = 'black'

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
            if (train_accuracy_name not in data.columns) or (val_accuracy_name not in data.columns) or (randomguess_name not in data.columns):
                continue

            most_important = alphas[d] == 1.0
            last = (d + 1) == len(datas)

            # define data_title
            data_title = csv_titles[d]

            # Plot the accuracy
            plt.plot(data[train_accuracy_name], label=f"{data_title} Training Accuracy", color=colors[d], alpha=alphas[d] * training_alpha_multiplier, linestyle='dashed')
            plt.plot(data[val_accuracy_name], label=f"{data_title} Validation Accuracy", color=colors[d], alpha=alphas[d], linestyle='solid')

            if last:
                plt.axhline(y=data[randomguess_name][0], linestyle='dashdot', label=f"Random Guess", color=randomguess_color, alpha=alphas[d])

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
            if (train_loss_name not in data.columns) or (val_loss_name not in data.columns):
                continue

            most_important = alphas[d] == 1.0
            last = (d + 1) == len(datas)

            # define data_title
            data_title = csv_titles[d]

            training_loss = data[train_loss_name]
            training_loss = training_loss[np.isfinite(training_loss)]

            validation_loss = data[val_loss_name]
            validation_loss = validation_loss[np.isfinite(validation_loss)]

            plt.plot(training_loss, label=f"{data_title} Training Loss", color=colors[d], alpha=alphas[d] * training_alpha_multiplier, linestyle='dashed')
            plt.plot(validation_loss, label=f"{data_title} Validation Loss", color=colors[d], alpha=alphas[d], linestyle='solid')

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


def make_confusion_matrix(model_path, tasktolabels_path, collapse_predictions=True, task='common:apparel_class', steps=3000, show_useless_scatter_plot=False):
    """

    :param model_path: Path in which the model resides.
    :param tasktolabels_path: The task-to-labels JSON that maps the integer predictions to label names.
    :param collapse_predictions: Whether a prediction should collapse into one-hot encoding (i.e. [0, 0, 1, 0]) or should be kept as a distribution (i.e. [0.1, 0.2, 0.6, 0.1])
    :param task: Which task to consider.
    :param steps: How many validation images should be used
    :return: None
    """
    with open(tasktolabels_path) as json_file:
        task_to_labels = json.load(json_file)

    task_aslist = [task]
    df = multitask.get_multitask_subset(data_type='validation',
                                        tasks=task_aslist,
                                        randomize_missing_labels=False,
                                        task_to_labels=task_to_labels,
                                        verbose=0)[0]
    generator = multitask.get_generators(None, df, task_aslist, batch_size=steps, shuffle=False)[1]

    model = keras.models.load_model(model_path)

    names = list(task_to_labels[task])
    C = len(names)

    # row is true, column is pred
    matrix = np.zeros(shape=(C, C))
    matrix_counts = np.zeros(shape=(C, C))

    valid_func = lambda x: x != -1

    for data_batch, labels_batch in generator:
        print("Getting data and labels...")
        labels_batch = labels_batch[0]

        # get % of each label
        p = ""
        presence = np.zeros(shape=C)
        for l in range(-1, C, 1):
            name = "missing"
            if l >= 0:
                name = names[l]

            m = np.mean(labels_batch == l)
            p = p + f"{name}: {m:.1%}; "

            if l >= 0:
                presence[l] = m

        print(p)

        labels = labels_batch
        data = data_batch

        break

    predicted_labels = model.predict(data)

    # the model may be trained for more than one task. find the right one
    # this can break if there is more than one task with the same amount of labels but that's a problem for future me.
    # or never me
    for t in range(len(predicted_labels)):
        if len(predicted_labels[t].shape) < 2: break

        if predicted_labels[t].shape[1] == C:
            predicted_labels = predicted_labels[t]
            break

    pred_df = pd.DataFrame()

    pred_df["y_true"] = labels

    # print(labels)
    # print(predicted_labels)

    pred_df["y_pred"] = [pred for pred in predicted_labels]
    pred_df["y_pred_sparse"] = pred_df["y_pred"].apply(lambda x: np.argmax(x))

    is_valid = pred_df["y_true"].apply(valid_func)

    pred_df = pred_df[is_valid]

    predicted_labels = pred_df["y_pred"]
    labels = pred_df["y_true"]

    def collapse_distribution(pred):
        sparse = np.argmax(pred)
        pred = np.zeros(shape=len(pred))
        pred[sparse] = 1.

        return pred

    if collapse_predictions:
        predicted_labels = predicted_labels.apply(collapse_distribution)

    for c in range(C):
        mask = labels == c

        if np.sum(mask) > 0:
            y_pred = predicted_labels[mask]
            y_pred = np.array(y_pred.to_list())

            matrix[c, :] = matrix[c, :] + np.mean(y_pred, axis=0)
            matrix_counts[c, :] = matrix_counts[c, :] + np.sum(y_pred, axis=0)

    names_true = pd.Series(data=names, name="True")
    names_pred = pd.Series(data=names, name="Prediction")
    result = pd.DataFrame(data=matrix, index=names_true, columns=names_pred)
    result_count = pd.DataFrame(data=matrix_counts, index=names_true, columns=names_pred, dtype=int)

    print("Confusion Matrix (Row-Normalized)")
    print(result)

    print("Confusion Matrix (Absolute counts)")
    print(result_count)

    print("Precision, recall and F1 scores:")

    precisions = np.zeros(shape=C)
    recalls = np.zeros(shape=C)
    f1scores = np.zeros(shape=C)

    precision_recall_df = pd.DataFrame(index=names, columns=["Precision", "Recall", "F1 Score"])

    for c in range(C):
        precisions[c] = matrix_counts[c, c] / np.sum(matrix_counts[:, c])
        recalls[c] = matrix_counts[c, c] / np.sum(matrix_counts[c, :])

        f1scores[c] = 2. * precisions[c] * recalls[c] / (precisions[c] + recalls[c])

        print(f"{names[c]}: Precision {precisions[c]:.1%}; Recall {recalls[c]:.1%}; F1 Score {f1scores[c]:.2f}")

    precision_recall_df["Precision"] = precisions
    precision_recall_df["Recall"] = recalls
    precision_recall_df["F1 Score"] = f1scores

    p = np.mean(precisions)
    r = np.mean(recalls)
    f = np.mean(f1scores)

    precision_recall_df.loc["Mean", ["Precision", "Recall", "F1 Score"]] = [p, r, f]

    # print(precision_recall_df)

    print("")
    print(f"Mean Precision: {p:.1%}; Mean Recall: {r:.1%}")

    print("")
    print(f"Overall F1 Score (Average of F1 Scores): {f:.2f}")

    print(f"Overall F1 Score (F1 Score of averages): {2. * p * r / (p + r):.2f}")

    for norm in [None]: # ('true', 'pred', None)
        values_format = ".0%"
        if norm is None:
            values_format = None

        confusion_matrix.from_predictions(y_true=pred_df["y_true"],
                                          y_pred=pred_df["y_pred_sparse"],
                                          normalize=None,
                                          labels=list(range(C)),
                                          display_labels=names,
                                          values_format=values_format,
                                          cmap='Blues',
                                          include_values=(C < 7))

    plt.show()

    if show_useless_scatter_plot:
        x = presence
        y = [matrix[c, c] for c in range(C)]

        plt.scatter(
            x = x,
            y = y,
            label="Points"
        )

        log_x = pd.Series(np.log(x))
        log_y = pd.Series(np.log(y))

        mask = ~log_x.isna() & ~log_y.isna() & pd.Series(x).apply(lambda u: u > 0) & pd.Series(y).apply(lambda u: u > 0)
        log_x = log_x[mask]
        log_y = log_y[mask]

        log_x = np.array(log_x.to_list()).reshape(-1, 1)
        log_y = np.array(log_y.to_list())

        reg = LinearRegression().fit(log_x, log_y)

        grid = np.linspace(start=0.05, stop=1, num=50).reshape(-1, 1)
        fittedline = np.exp(reg.predict(np.log(grid)))

        plt.plot(grid, fittedline, label="Best Fit", alpha=0.5, linestyle="--")

        plt.title("Presence in dataset VS Accuracy")
        plt.xlabel("Presence")
        plt.xlim((0., 1.))
        xlocs, xlabels = plt.xticks()
        xlocs = np.arange(start=0., stop=1., step=0.2)
        xlabels = [f"{loc:.0%}" for loc in xlocs]
        plt.xticks(xlocs, xlabels)

        plt.ylabel("Accuracy")
        plt.ylim((0., 1.))
        ylocs, ylabels = plt.yticks()
        ylocs = np.arange(start=0., stop=1., step=0.2)
        ylabels = [f"{loc:.0%}" for loc in ylocs]
        plt.yticks(ylocs, ylabels)
        plt.legend()

        plt.show()

    return result_count


"""
csv_path = 'shoeMultiTaskModel_trainingLog (3).csv'
tasks=('shoe:up height', 'shoe:age', 'shoe:color') 
make_training_log_plots(csv_path, tasks)
"""
