import numpy as np
import matplotlib.pyplot as plt


def _rearrange(confusion_matrix):
    new_order = [-1] * np.min(confusion_matrix.shape)
    # Find best possible diagonal by using maximum values in the confusion matrix
    hits_sorted = np.unravel_index(np.argsort(confusion_matrix, axis=None)[::-1], confusion_matrix.shape)
    i = 0
    while -1 in new_order:
        row = hits_sorted[0][i]
        column = hits_sorted[1][i]
        # Check if column has already been specified
        if new_order[row] == -1 and column not in new_order:
            new_order[row] = column
        i += 1
    # If there are more columns than rows, order remaining columns by highest value
    if confusion_matrix.shape[1] > confusion_matrix.shape[0]:
        missing_columns = np.array(list(set(range(confusion_matrix.shape[1])) - set(new_order)))
        missing_order = np.argsort(np.max(confusion_matrix[:, missing_columns], axis=0))[::-1]
        new_order += missing_columns[missing_order].tolist()
    # Overwrite confusion matrix
    new_confusion_matrix = confusion_matrix[:, new_order]
    return new_confusion_matrix


def _plot_confusion_matrix(confusion_matrix, show_text=True, figsize=(10, 10), cmap="YlGn", textcolor="black", vmin=None,
                           vmax=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(confusion_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    if show_text:
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, confusion_matrix[i, j],
                        ha="center", va="center", color=textcolor)
    plt.show()


class ConfusionMatrix():

    def __init__(self, labels_true, labels_pred):
        self.true_clusters = np.unique(labels_true)
        self.pred_clusters = np.unique(labels_pred)
        conf_matrix = np.zeros((len(self.true_clusters), len(self.pred_clusters)), dtype=int)
        for i, gt_label in enumerate(self.true_clusters):
            # Get predictions which should be labeled with corresponding gt label
            point_labels = labels_pred[labels_true == gt_label]
            # Get different prediction labels
            labels, cluster_sizes = np.unique(point_labels, return_counts=True)
            for j, pred_label in enumerate(labels):
                conf_matrix[i, np.argwhere(self.pred_clusters == pred_label)[0][0]] = cluster_sizes[j]
        self.confusion_matrix = conf_matrix

    def __str__(self):
        return str(self.confusion_matrix)

    def rearrange(self):
        new_confusion_matrix = _rearrange(self.confusion_matrix)
        self.confusion_matrix = new_confusion_matrix

    def plot(self, show_text=True, figsize=(10, 10), cmap="YlGn", textcolor="black", vmin=0, vmax=None):
        _plot_confusion_matrix(self.confusion_matrix, show_text, figsize, cmap, textcolor, vmin, vmax)
