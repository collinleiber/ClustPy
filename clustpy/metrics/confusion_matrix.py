import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from clustpy.metrics._metrics_utils import _check_labels_arrays


def _rearrange(confusion_matrix: np.ndarray) -> (np.ndarray, np.ndarray):
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
        The rearranged confusion matrix
        (If number of ground truth labels is larger than the number of predicted labels, the resulting confusion matrix will be quadradic with multiple 0 columns),
        The indices regarding the rearrangement
    """
    # Change order using the Hungarian Method
    max_number_labels = max(confusion_matrix.shape)
    rearranged_confusion_matrix = np.zeros((max_number_labels, max_number_labels), dtype=confusion_matrix.dtype)
    # Linear sum assignment tries to minimize the diagonal sum -> use negative confusion_matrix
    rearranged_confusion_matrix[:confusion_matrix.shape[0], :confusion_matrix.shape[1]] = confusion_matrix
    indices = linear_sum_assignment(-rearranged_confusion_matrix)
    # Change order of the columns
    rearranged_order = indices[1]
    rearranged_confusion_matrix = rearranged_confusion_matrix[:, rearranged_order]
    rearranged_confusion_matrix = rearranged_confusion_matrix[:confusion_matrix.shape[0], :]
    # If there are more columns than rows sort remaining columns by highest value
    if confusion_matrix.shape[1] > confusion_matrix.shape[0]:
        missing_columns = np.arange(confusion_matrix.shape[0], confusion_matrix.shape[1])
        missing_columns_order = np.argsort(np.max(rearranged_confusion_matrix[:, missing_columns], axis=0))[::-1]
        rearranged_confusion_matrix[:, missing_columns] = rearranged_confusion_matrix[:, missing_columns[missing_columns_order]]
        rearranged_order[missing_columns] = rearranged_order[missing_columns[missing_columns_order]]
    return rearranged_confusion_matrix, rearranged_order


def _plot_confusion_matrix(confusion_matrix: np.ndarray, show_text: bool, row_names : list, column_names : list, figsize: tuple, cmap: str, textcolor: str,
                           vmin: float, vmax: float) -> None:
    """
    Plot the confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The confusion matrix to plot
    show_text : bool
        Show the value in each cell as text
    row_names : list
        List of containing the names of the rows
    column_names : list
        List of containing the names of the columns
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
    if len(row_names) != confusion_matrix.shape[0]:
        raise ValueError("Length of the ground_truth_names must match the number of rows (ground turth clusters) in the confusion matrix. Length is {0} and number of rows is {1}".format(row_names, confusion_matrix.shape[0]))
    if len(column_names) != confusion_matrix.shape[1]:
        raise ValueError("Length of the ground_truth_names must match the number of rows (ground turth clusters) in the confusion matrix. Length is {0} and number of rows is {1}".format(column_names, confusion_matrix.shape[1]))
    fig, ax = plt.subplots(figsize=figsize)
    # Plot confusion matrix using colors
    ax.imshow(confusion_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(confusion_matrix.shape[1]))
    ax.set_xticklabels(column_names)
    ax.set_yticks(np.arange(confusion_matrix.shape[0]))
    ax.set_yticklabels(row_names)
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
    shape : tuple | str | None
        The desired shape of the confusion matrix. 
        Can be "square" to encforce a squared confusion matrix (default: None)

    Attributes
    ----------
    confusion_matrix : np.ndarray
        The confusion matrix
    """

    def __init__(self, labels_true: np.ndarray, labels_pred: np.ndarray, shape: tuple | str | None=None):
        labels_true, labels_pred = _check_labels_arrays(labels_true, labels_pred)
        true_clusters, true_clusters_idx = np.unique(labels_true, return_inverse=True)
        pred_clusters, pred_clusters_idx = np.unique(labels_pred, return_inverse=True)
        self.true_clusters = true_clusters
        self.pred_clusters = pred_clusters
        if shape is None:
            conf_matrix = np.zeros((len(true_clusters), len(pred_clusters)), dtype=int)
        else:
            if shape == "square":
                max_labels = max(len(true_clusters), len(pred_clusters))
                shape = (max_labels, max_labels)
            else:
                assert len(shape) == 2 and shape[0] >= len(true_clusters) and shape[1] >= len(pred_clusters), f"Shape must be 'square' or a tuple containing two values such that shape[0] >= len(np.unique(labels_true)) and shape[1] >= len(np.unique(labels_pred)). Your values: shape = {shape}, len(np.unique(labels_true)) = {len(np.unique(labels_true))}, len(np.unique(labels_pred)) = {len(np.unique(labels_pred))}"
            conf_matrix = np.zeros(shape, dtype=int)
            # Fill unique label information (self.true_clusters and self.pred_clusters) with -2 placeholders
            if shape[0] > len(true_clusters):
                self.true_clusters = np.append(self.true_clusters, [-2] * (shape[0] - len(true_clusters)))
            if shape[1] > len(pred_clusters):
                self.pred_clusters = np.append(self.pred_clusters, [-2] * (shape[1] - len(pred_clusters)))
        np.add.at(conf_matrix, (true_clusters_idx, pred_clusters_idx), 1)
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
            Should the new confusion matrix overwrite the original one (default: True)

        Returns
        -------
        rearranged_confusion_matrix : np.ndarray
            The rearranged confusion matrix
            If number of ground truth labels is larer than the number of predicted labels, the resulting confusion matrix will be quadradic with multiple 0 columns.
        """
        rearranged_confusion_matrix, rearranged_order = _rearrange(self.confusion_matrix)
        if inplace:
            self.confusion_matrix = rearranged_confusion_matrix
            self.pred_clusters = self.pred_clusters[rearranged_order[:len(self.pred_clusters)]]
        return rearranged_confusion_matrix

    def plot(self, show_text: bool = True, ground_truth_names: list | None = None, 
             figsize: tuple = (10, 10), cmap: str = "YlGn", textcolor: str = "black", 
             vmin: int = 0, vmax: int = None) -> None:
        """
        Plot the confusion matrix.

        Parameters
        ----------
        show_text : bool
            Show the value in each cell as text (default: True)
        ground_truth_names : list | None
            List of containing the names of the ground truth clusters
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
        if ground_truth_names is None:
            ground_truth_names = self.true_clusters
        _plot_confusion_matrix(self.confusion_matrix, show_text, ground_truth_names, self.pred_clusters, figsize, cmap, textcolor, vmin, vmax)
