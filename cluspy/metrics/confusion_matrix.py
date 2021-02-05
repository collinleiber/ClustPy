import numpy as np
import matplotlib.pyplot as plt


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
        new_order = [-1] * np.min(self.confusion_matrix.shape)
        # Find best possible diagonal by using maximum values in the confusion matrix
        hits_sorted = np.unravel_index(np.argsort(self.confusion_matrix, axis=None)[::-1], self.confusion_matrix.shape)
        i = 0
        while -1 in new_order:
            row = hits_sorted[0][i]
            column = hits_sorted[1][i]
            # Check if column has already been specified
            if new_order[row] == -1 and column not in new_order:
                new_order[row] = column
            i += 1
        # If there are more columns than rows, order remaining columns by highest value
        if self.confusion_matrix.shape[1] > self.confusion_matrix.shape[0]:
            missing_columns = np.array(list(set(range(self.confusion_matrix.shape[1])) - set(new_order)))
            missing_order = np.argsort(np.max(self.confusion_matrix[:, missing_columns], axis=0))[::-1]
            new_order += missing_columns[missing_order].tolist()
        # Overwrite confusion matrix
        self.confusion_matrix = self.confusion_matrix[:, new_order]

    def plot(self, figsize=(10, 10), cmap="YlGn", textcolor="black"):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.confusion_matrix, cmap=cmap)
        for i in range(len(self.true_clusters)):
            for j in range(len(self.pred_clusters)):
                ax.text(j, i, self.confusion_matrix[i, j],
                        ha="center", va="center", color=textcolor)
        plt.show()
