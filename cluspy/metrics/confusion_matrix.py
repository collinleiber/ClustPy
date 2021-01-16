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

    def plot(self, figsize=(10, 10), cmap="YlGn", textcolor="black"):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.confusion_matrix, cmap=cmap)
        for i in range(len(self.true_clusters)):
            for j in range(len(self.pred_clusters)):
                ax.text(j, i, self.confusion_matrix[i, j],
                        ha="center", va="center", color=textcolor)
        plt.show()
