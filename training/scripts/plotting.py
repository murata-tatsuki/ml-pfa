"""Loss / cluster-space plotting helpers (development use)."""

from __future__ import annotations

import matplotlib.pylab as plt
import numpy as np


def colorlabel(y, label):
    unique_label = np.unique(label)
    if y == unique_label[0]:
        return "b"
    if y == unique_label[1]:
        return "g"


def check_plots(coords_list, data_y_list):
    coords_list = np.array(coords_list[0])
    label = np.array(data_y_list[0][0:4000])
    _, ax = plt.subplots(figsize=(8, 6))
    for x1, y1, label1 in zip(
        coords_list[0:4000, 0], coords_list[0:4000, 1], label
    ):
        ax.scatter(x1, y1, c=colorlabel(label1, label))
    plt.show()


def plot_history(train_loss_history, test_loss_history):
    if isinstance(test_loss_history, list):
        plt.figure(figsize=(8, 6))
        plt.plot(test_loss_history, label="test_loss", lw=3, c="b")
        plt.plot(train_loss_history, label="train_loss", lw=3, c="green")
        plt.title("loss function")
        plt.legend(fontsize=14)
        plt.show()


def plot_acc_history(train_acc_history, test_acc_history):
    if isinstance(test_acc_history, list):
        plt.figure(figsize=(8, 6))
        plt.plot(test_acc_history, label="test_acc", lw=3, c="b")
        plt.plot(train_acc_history, label="train_acc", lw=3, c="green")
        plt.title("accuracy")
        plt.legend(fontsize=14)
        plt.show()
