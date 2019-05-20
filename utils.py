#!/usr/bin/env python3

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_confusion_matrix(y_actu, y_pred, norm=True, title='Confusion matrix', cmap=plt.cm.Blues):
    df_confusion = pd.crosstab(y_actu, y_pred.reshape(y_pred.shape[0],), rownames=['Actual'], colnames=['Predicted'], margins=True)
    if norm == True:
        df_confusion = df_confusion / df_confusion.sum(axis=1)
    plt.figure(figsize=(15, 7))
    plt.matshow(df_confusion, fignum=1, cmap=cmap, norm=colors.PowerNorm(gamma=0.2))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    for (i, j), z in np.ndenumerate(df_confusion):
        z_str = '{:1.0f}'.format(z)
        if norm == True:
            z_str = '{:0.2f}'.format(z)
        plt.text(j, i, z_str, ha='center', va='center') 