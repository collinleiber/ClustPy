from unittest.mock import patch
from clustpy.utils import plot_with_transformation, plot_1d_data, plot_2d_data, plot_3d_data, plot_histogram, plot_image, \
    plot_scatter_matrix
import numpy as np
from sklearn.decomposition import PCA, FastICA
from clustpy.data import load_nrletters, load_optdigits
import matplotlib.pyplot as plt


def _get_data_labels_centers():
    X = np.array([[1, 2, 3, 4, 5], [2, 2, 2, 3, 3], [1.4, 1.5, 1.6, 1.7, 1.8], [2, 3, 2, 3, 2],
                  [1.1, 1.0, 1.1, 1.0, 1.2],
                  [10, 10, 9, 11, 10.5], [11, 12, 11, 11, 10.1], [11.5, 11.4, 11.5, 11.3, 11.3], [12, 12, 12, 11, 12],
                  [12.5, 12.3, 12.4, 12.3, 12.5], [13, 13.1, 12, 13.2, 13.3], [14, 14, 15, 14, 16],
                  [15, 15.1, 15.1, 15.2, 14.7], [15.5, 15.6, 15.6, 15.5, 17], [15.7, 15.6, 14.6, 14.7, 15.4]])
    L = np.array([0] * 5 + [1] * 7 + [2] * 3)
    centers = np.array([np.mean(X[L == i], axis=0) for i in range(3)])
    L_true = np.array([0] * 5 + [1] * 10)
    return X, L, centers, L_true


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_with_transformation(mock_fig):
    X, L, centers, L_true = _get_data_labels_centers()
    assert None == plot_with_transformation(X, L, centers, L_true, 1, PCA, show_legend=True, scattersize=11, equal_axis=True, title="test", show_plot=True)
    assert None == plot_with_transformation(X, L, centers, L_true, 2, PCA, show_legend=True, scattersize=11, equal_axis=True, title="test", show_plot=True)
    assert None == plot_with_transformation(X, L, centers, L_true, 3, FastICA, show_legend=True, scattersize=11, equal_axis=True, title="test", show_plot=True)
    # Check if this works ('reduced' dim > data dim)
    assert None == plot_with_transformation(X, L, centers, L_true, X.shape[1] + 1, FastICA, show_legend=True, scattersize=11, equal_axis=True, title="test", show_plot=True)


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_1d_data(mock_fig):
    X, L, centers, L_true = _get_data_labels_centers()
    assert None == plot_1d_data(X[:, 0], L, centers[:, 0], L_true, show_legend=True, title="test", show_plot=True)


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_2d_data(mock_fig):
    X, L, centers, L_true = _get_data_labels_centers()
    assert None == plot_2d_data(X[:, :2], L, centers[:, :2], L_true, show_legend=True, scattersize=11, equal_axis=True, title="test", show_plot=True)


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_3d_data(mock_fig):
    X, L, centers, L_true = _get_data_labels_centers()
    assert None == plot_3d_data(X[:, :3], L, centers[:, :3], L_true, show_legend=True, scattersize=11, title="test", show_plot=True)
    plt.figure()  # Create new figure for future plots


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_histogram(mock_fig):
    X, L, centers, L_true = _get_data_labels_centers()
    assert None == plot_histogram(X[:, 0], L, density=True, n_bins=50, show_legend=True, title="test", show_plot=True)


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_image_using_optdigits_and_nrletters(mock_fig):
    X = load_optdigits().data
    assert None == plot_image(X[0], True, (8, 8), title="test", show_plot=True)
    dataset = load_nrletters()
    assert None == plot_image(dataset.data[0], False, (9, 7, 3), is_color_channel_last=True, max_value=255, min_value=0, title="test", show_plot=True)
    assert None == plot_image(dataset.images[0], False, (9, 7, 3), is_color_channel_last=False, max_value=255, min_value=0, title="test", show_plot=True)


@patch("matplotlib.pyplot.show")  # Used to test plots (show will not be called)
def test_plot_scatter_matrix(mock_fig):
    X, L, centers, L_true = _get_data_labels_centers()
    plot_scatter_matrix(X, L, centers, L_true, density=True, n_bins=50, show_legend=True, scattersize=11, equal_axis=True, max_dimensions=12,
                        title="test", show_plot=True)
    # Only check if error is thrown
    assert True
