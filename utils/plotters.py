import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
import numpy as np


def plot_error(file=None):
    with open(file=file) as error:
        iter(error)
        errors = list(error)
        plt.title('Model Training Error')
        plt.plot([float(error) for error in errors])
        plt.show()


def plot_confusion(y_true=None, y_pred=None, title='Confusion Matrix', fmt='d', labels=None):
    cm = {'Actual': y_true, 'Predicted': y_pred}
    cm = pd.DataFrame(cm, columns=cm.keys())
    confusion_matrix = pd.crosstab(cm['Actual'], cm['Predicted'], rownames=['Actual'], colnames=['Predicted'])
    sb.heatmap(confusion_matrix, cmap='Oranges', annot=True, fmt=fmt, robust=True, yticklabels=labels, cbar=False, annot_kws={'size': 14})
    plt.title(title)
    plt.show()


def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color='red', linestyle='--', label='mean')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--', label='SD +1')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--', label='SD -1')
    plt.xlabel('Average Between Observer and Model')
    plt.ylabel('Difference between TCD Truth and TCD Predictions')
    plt.legend()
    plt.title('Bland-Altman Plot of TCD agreement')
    plt.show()
