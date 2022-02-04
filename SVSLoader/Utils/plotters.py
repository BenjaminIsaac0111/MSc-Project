import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np

COMPONENTS = ('0: Non-Inf',
              '1: Tumour',
              '2: Str/Fib',
              '3: Necr',
              '4: Vessel',
              '5: Infl',
              '6: TumLu',
              '7: Mucin',
              '8: Muscle')


def plot_error(file=None):
    with open(file=file) as error:
        iter(error)
        errors = list(error)
        plt.title('Model Training Error')
        plt.plot([float(error) for error in errors])
        plt.show()


def plot_confusion(y_true=None, y_pred=None, title='Confusion Matrix', fmt='d', labels=COMPONENTS):
    cm = {'Actual': y_true, 'Predicted': y_pred}
    cm = pd.DataFrame(cm, columns=cm.keys())
    confusion_matrix = pd.crosstab(cm['Actual'], cm['Predicted'], rownames=['Actual'], colnames=['Predicted'])
    sb.heatmap(confusion_matrix, cmap='Oranges', annot=True, fmt=fmt, robust=True, yticklabels=labels, cbar=False,
               annot_kws={'size': 14})
    plt.title(title)
    plt.show()


def accuracy_per_image_plot(accuracies=None, xlabel='Accuracy', ylabel='Per Image Count', title=None, bins=20):
    md = np.mean(accuracies)
    sd = np.std(accuracies, axis=0)
    plt.hist(accuracies, bins=20, histtype='stepfilled')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axvline(md, color='red', linestyle='--', label='mean')
    plt.axvline(md + 1.96 * sd, color='gray', linestyle='--', label='SD +1')
    plt.axvline(md - 1.96 * sd, color='gray', linestyle='--', label='SD -1')
    plt.legend()
    plt.show()
