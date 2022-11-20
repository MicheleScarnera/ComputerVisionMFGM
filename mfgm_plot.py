
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

def make_training_log_plots(csv_path, tasks):
    """

    :param csv_path: path to csv
    :param tasks: for example ['shoe:up height', 'shoe:age', 'shoe:color']. it will make plots of these tasks (and of the overall loss/accuracy metrics)
    :return: """
    
    data = pd.read_csv(csv_path)
    
    for i in range(len(tasks)):
        task = tasks[i]
        
        # Plot the accuracy
        plt.plot(data[task + '_accuracy'])
        plt.plot(data['val_'+task + '_accuracy'])
        plt.plot(data[task + '_randomguess'])
        plt.title(task)
        plt.xlabel('epochs')
        plt.legend(['Accuracy ', 'Validation Accuracy', 'Random Guess'])
        plt.xticks(range(0, len(data['epoch']),2))
        plt.show()
        
        # Plot the loss
        plt.plot(data[task + '_loss'])
        plt.plot(data['val_'+task + '_loss'])
        plt.title(task)
        plt.xlabel('epochs')
        plt.legend(['Loss ', 'Validation Loss'])
        plt.xticks(range(0, len(data['epoch']),2))
        plt.show()
        
    return

csv_path = 'shoeMultiTaskModel_trainingLog (3).csv'
tasks=('shoe:up height', 'shoe:age', 'shoe:color') 
make_training_log_plots(csv_path, tasks)        