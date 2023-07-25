import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def _rearrange(confusion_matrix: np.ndarray) -> np.ndarray:
    """
    Rearrange the confusion matrix in such a way that the sum of the diagonal is maximized.
    Thereby, the best matching combination of labels will be shown.
    Uses the Hungarian Method to identify the best match.
    If parameter inplace is set to True, this method will change the original confusion matrix.
    Else the rearranged matrix will only be returned.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The original confusion matrix

    Returns
    -------
    rearranged_confusion_matrix : np.ndarray
        The rearranged confusion matrix.
        If number of ground truth labels is larger than the number of predicted labels, the resulting confusion matrix will be quadradic with multiple 0 columns.
    """
    # Change order using the Hungarian Method
    max_number_labels = max(confusion_matrix.shape)
    rearranged_confusion_matrix = np.zeros((max_number_labels, max_number_labels), dtype=int)
    # Linear sum assignment tries to minimize the diagonal sum -> use negative confusion_matrix
    rearranged_confusion_matrix[:confusion_matrix.shape[0], :confusion_matrix.shape[1]] = -confusion_matrix
    indices = linear_sum_assignment(rearranged_confusion_matrix)
    # Revert values back to positive range, change order of the columns
    rearranged_confusion_matrix = -rearranged_confusion_matrix[:, indices[1]]
    rearranged_confusion_matrix = rearranged_confusion_matrix[:confusion_matrix.shape[0], :]
    # If there are more columns than rows sort remaining columns by highest value
    if confusion_matrix.shape[1] > confusion_matrix.shape[0]:
        missing_columns = np.arange(confusion_matrix.shape[0], confusion_matrix.shape[1])
        missing_columns_order = np.argsort(np.max(rearranged_confusion_matrix[:, missing_columns], axis=0))[::-1]
        rearranged_confusion_matrix[:, missing_columns] = rearranged_confusion_matrix[:, missing_columns[missing_columns_order]]
    return rearranged_confusion_matrix


def _plot_confusion_matrix(confusion_matrix: np.ndarray, show_text: bool, figsize: tuple, cmap: str, textcolor: str,
                           vmin: float, vmax: float) -> None:
    """
    Plot the confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The confusion matrix to plot
    show_text : bool
        Show the value in each cell as text
    figsize : tuple
        Tuple indicating the height and width of the plot
    cmap : str
        Colormap used for the plot
    textcolor : str
        Color of the text. Only relevant if show_text is True
    vmin : float
        Minimum possible value within a cell of the confusion matrix.
        If None, it will be set as the minimum value within the confusion matrix.
        Used to choose the color from the colormap
    vmax : float
        Maximum possible value within a cell of the confusion matrix.
        If None, it will be set as the maximum value within the confusion matrix.
        Used to choose the color from the colormap
    """
    fig, ax = plt.subplots(figsize=figsize)
    # Plot confusion matrix using colors
    ax.imshow(confusion_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    # Optional: Add text to the color cells
    if show_text:
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, confusion_matrix[i, j],
                        ha="center", va="center", color=textcolor)
    plt.show()


class ConfusionMatrix():
    """
    Create a Confusion Matrix of predicted and ground truth labels.
    Each row corresponds to a ground truth label and each column to a predicted label.
    The number in each cell (i, j) indicates how many objects with ground truth label i have been predicted label j.

    Parameters
    ----------
    labels_true : np.ndarray
        The ground truth labels of the data set
    labels_pred : np.ndarray
        The labels as predicted by a clustering algorithm

    Attributes
    ----------
    confusion_matrix : np.ndarray
        The confusion matrix
    """

    def __init__(self, labels_true: np.ndarray, labels_pred: np.ndarray):
        assert labels_true.shape[0] == labels_pred.shape[0], "Number of true and predicted labels must match"
        self.true_clusters = np.unique(labels_true)
        self.pred_clusters = np.unique(labels_pred)
        conf_matrix = np.zeros((self.true_clusters.shape[0], self.pred_clusters.shape[0]), dtype=int)
        for i, gt_label in enumerate(self.true_clusters):
            # Get predictions which should be labeled with corresponding gt label
            point_labels = labels_pred[labels_true == gt_label]
            # Get different prediction labels
            labels, cluster_sizes = np.unique(point_labels, return_counts=True)
            for j, pred_label in enumerate(labels):
                conf_matrix[i, np.where(self.pred_clusters == pred_label)[0][0]] = cluster_sizes[j]
        self.confusion_matrix = conf_matrix

    def __str__(self):
        """
        Print the confusion matrix.

        Returns
        -------
        str_confusion_matrix : str
            The confusion matrix as a string
        """
        str_confusion_matrix = str(self.confusion_matrix)
        return str_confusion_matrix

    def rearrange(self, inplace: bool = True) -> np.ndarray:
        """
        Rearrange the confusion matrix in such a way that the sum of the diagonal is maximized.
        Thereby, the best matching combination of labels will be shown.
        Uses the Hungarian Method to identify the best match.
        If parameter inplace is set to True, this method will change the original confusion matrix.
        Else the rearranged matrix will only be returned.

        Parameters
        ----------
        inplace : bool
            Should the new confusion matrix overwrite the original one

        Returns
        -------
        rearranged_confusion_matrix : np.ndarray
            The rearranged confusion matrix
            If number of ground truth labels is larer than the number of predicted labels, the resulting confusion matrix will be quadradic with multiple 0 columns.
        """
        rearranged_confusion_matrix = _rearrange(self.confusion_matrix)
        if inplace:
            self.confusion_matrix = rearranged_confusion_matrix
        return rearranged_confusion_matrix

    def plot(self, show_text: bool = True, figsize: tuple = (10, 10), cmap: str = "YlGn", textcolor: str = "black",
             vmin: int = 0, vmax: int = None) -> None:
        """
        Plot the confusion matrix.

        Parameters
        ----------
        show_text : bool
            Show the value in each cell as text (default: True)
        figsize : tuple
            Tuple indicating the height and width of the plot (default: (10, 10))
        cmap : str
            Colormap used for the plot (default: "YlGn")
        textcolor : str
            Color of the text. Only relevant if show_text is True (default: "black")
        vmin : int
            Minimum possible value within a cell of the confusion matrix.
            If None, it will be set as the minimum value within the confusion matrix.
            Used to choose the color from the colormap (default: 0)
        vmax : int
            Maximum possible value within a cell of the confusion matrix.
            If None, it will be set as the maximum value within the confusion matrix.
            Used to choose the color from the colormap (default: None)
        """
        _plot_confusion_matrix(self.confusion_matrix, show_text, figsize, cmap, textcolor, vmin, vmax)
