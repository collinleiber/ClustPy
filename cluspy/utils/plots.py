import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats
import matplotlib.patches as mpatches

# Circle, Square, Diamond, Plus, X, Triangle down, Star, Pentagon, Triangle Up, Triangle left, Triangle right, Hexagon
MARKERS = ["o", "s", "D", "P", "X", "v", "*", "p", "^", ">", "<", "h"]

_MIN_OBJECTS_FOR_DENS_PLOT = 3


def plot_with_transformation(X, labels=None, centers=None, true_labels=None, plot_dimensionality=2,
                             transformation_class=PCA, scattersize=10, equal_axis=False, show_plot=True):
    assert plot_dimensionality > 0, "Plot dimensionality must be > 0"
    if X.ndim == 1:
        plot_dimensionality = 1
    elif plot_dimensionality > X.shape[1]:
        print(
            "[WARNING] plot_dimensionality ({0}) is higher than the dimensionaliyty of the input dataset ({1}). "
            "plot_dimensionality will therefore be set to {1}.".format(
                plot_dimensionality, X.shape[1]))
        plot_dimensionality = X.shape[1]
    elif plot_dimensionality < X.shape[1]:
        # Check if transformation dimensionality is smaller than number of features
        trans = transformation_class(n_components=plot_dimensionality)
        X = trans.fit_transform(X)
        if centers is not None:
            centers = trans.transform(centers)
    if true_labels is not None:
        unique_true_labels = np.unique(true_labels)
    if plot_dimensionality == 1:
        # 1d Plot
        plot_1d_data(X, labels=labels, centers=centers, true_labels=true_labels, show_plot=False)
    elif plot_dimensionality == 2:
        # 2d Plot
        if true_labels is None:
            plt.scatter(X[:, 0], X[:, 1], c=labels, s=scattersize)
        else:
            # Change marker for true labels
            for lab_index, true_lab in enumerate(unique_true_labels):
                marker = MARKERS[lab_index % len(MARKERS)]
                plt.scatter(X[true_labels == true_lab, 0], X[true_labels == true_lab, 1], s=scattersize,
                            c=labels if labels is None else labels[true_labels == true_lab], marker=marker,
                            vmin=np.min(labels), vmax=np.max(labels))
        if centers is not None:
            plt.scatter(centers[:, 0], centers[:, 1], s=scattersize * 1.5, color="red", marker="s")
            for j in range(len(centers)):
                plt.text(centers[j, 0], centers[j, 1], str(j), weight="bold")
        if equal_axis:
            plt.axis("equal")
    elif plot_dimensionality == 3:
        # 3d Plot
        fig = plt.figure()
        ax = Axes3D(fig)  # fig.add_subplot(111, projection='3d')
        if true_labels is None:
            ax.scatter(X[:, 0], X[:, 1], zs=X[:, 2], zdir='z', s=scattersize, c=labels, alpha=0.8)
        else:
            # Change marker for true labels
            for lab_index, true_lab in enumerate(unique_true_labels):
                marker = MARKERS[lab_index % len(MARKERS)]
                plt.scatter(X[true_labels == true_lab, 0], X[true_labels == true_lab, 1],
                            zs=X[true_labels == true_lab, 2], zdir='z', s=scattersize,
                            c=labels if labels is None else labels[true_labels == true_lab],
                            marker=marker, vmin=np.min(labels), vmax=np.max(labels), alpha=0.8)
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], zs=centers[:, 2], zdir='z', s=scattersize * 1.5, color="red",
                       marker="s")
            for j in range(len(centers)):
                ax.text(centers[j, 0], centers[j, 1], centers[j, 2], str(j), weight="bold")
    else:
        plot_scatter_matrix(X, labels=labels, centers=centers, true_labels=true_labels, scattersize=scattersize,
                            equal_axis=equal_axis, show_plot=False)
    if show_plot:
        plt.show()


def plot_1d_data(X, labels=None, centers=None, true_labels=None, show_plot=True):
    assert X.ndim == 1 or X.shape[1] == 1, "Data must be 1-dimensional"
    assert centers is None or centers.ndim == 1 or centers.shape[1] == 1, "Centers must be 1-dimensional"
    # Optional: Get first column of data
    if X.ndim == 2:
        X = X[:, 0]
    # fig, ax = plt.subplots(figsize=figsize)
    min_value = np.min(X)
    max_value = np.max(X)
    plt.hlines(1, min_value, max_value)  # Draw a horizontal line
    y = np.ones(len(X))
    plt.scatter(X, y, marker='|', s=500, c=labels)  # Plot a line at each location specified in X
    if centers is not None:
        # Optional: Get first column of centers
        if centers.ndim == 2:
            centers = centers[:, 0]
        yc = np.ones(len(centers))
        plt.scatter(centers, yc, s=300, color="red", marker="x")
        # plot one center text above line and next below ...
        centers_order = np.argsort(centers)
        centers_order = np.argsort(centers_order)
        for j in range(len(centers)):
            yt = 1.0005 if centers_order[j] % 2 == 0 else 0.9994
            plt.text(centers[j], yt, str(j), weight="bold")
    if true_labels is not None:
        plt.hlines(1.001, min_value, max_value)
        y_true = np.ones(len(X)) * 1.001
        plt.scatter(X, y_true, marker='|', s=500, c=true_labels)
    if show_plot:
        plt.show()


def plot_image(img_data: np.ndarray, black_and_white: bool = False, image_shape: tuple = None, max_value: float = None,
               min_value: float = None, show_plot: bool = True):
    """
    Expects a color image to occur in the HWC representation (height, width, color channels).

    Parameters
    ----------
    img_data : np.ndarray
    black_and_white : bool
    image_shape : tuple
        (height, width) for grayscale images or (height, width, number of channels) for color images
    max_value : float
        maximum pixel value, used for min-max normalization. Is often 255, if None the maximum value in the data set will be used (default: None)
    min_value : float
        maximum pixel value, used for min-max normalization. Is often 0, if None the minimum value in the data set will be used (default: 255)
    show_plot : bool

    Returns
    -------

    """
    assert img_data.ndim <= 3, "Image data can not have more than 3 dimensions."
    # Data range must match float between [0..1] or int between [0..255] -> use min-max transform
    if max_value is None:
        max_value = np.max(img_data)
    if min_value is None:
        min_value = np.min(img_data)
    img_data = (img_data - min_value) / (max_value - min_value)
    # Reshape array data
    if img_data.ndim == 1:
        img_data = img_data.reshape(image_shape)
    # Plot original image or a black-and-white version
    if black_and_white:
        plt.imshow(img_data, cmap="Greys")
    else:
        plt.imshow(img_data)
    plt.axis('off')
    if show_plot:
        plt.show()


def plot_histogram(X, labels=None, density=True, n_bins=100, show_legend=True, container=plt, show_plot=True):
    assert X.ndim == 1, "Data must be 1-dimensional"
    if labels is not None:
        unique_labels = np.unique(labels)
        # Manage colormap
        cmap = cm.get_cmap('viridis', 12)
        norm = Normalize(vmin=unique_labels[0], vmax=unique_labels[-1])
    # Plot histogram
    if labels is not None:
        for lab in unique_labels:
            # Get common label colors for histogram and density
            hist_color = cmap(norm(lab))
            container.hist(X[labels == lab], alpha=0.5, bins=n_bins, color=hist_color, range=(np.min(X), np.max(X)))
    else:
        container.hist(X, alpha=0.5, bins=n_bins, range=(np.min(X), np.max(X)))
    # Plot densities
    if density:
        # Histogram and density should share same x-axis
        twin_axis = container.twinx()
        twin_axis.yaxis.set_visible(False)
        if labels is not None:
            for lab in unique_labels:
                den_objects = X[labels == lab]
                if den_objects.shape[0] >= _MIN_OBJECTS_FOR_DENS_PLOT:
                    hist_color = cmap(norm(lab))
                    kde = stats.gaussian_kde(den_objects)
                    steps = np.linspace(np.min(den_objects), np.max(den_objects), 1000)
                    twin_axis.plot(steps, kde(steps), color=hist_color)
        elif X.shape[0] >= _MIN_OBJECTS_FOR_DENS_PLOT:
            kde = stats.gaussian_kde(X)
            steps = np.linspace(np.min(X), np.max(X), 1000)
            twin_axis.plot(steps, kde(steps))
    if show_legend and labels is not None:
        _add_legend(container, unique_labels, cmap, norm)
    if show_plot:
        plt.show()


def plot_scatter_matrix(X, labels=None, centers=None, true_labels=None, density=True, n_bins=100, show_legend=True,
                        scattersize=10, equal_axis=False, max_dimensions=10, show_plot=True):
    if X.shape[1] > max_dimensions:
        print(
            "[WARNING] Dimensionality of the dataset is larger than 10. Creation of scatter matrix plot will be aborted.")
        return
    # For single dimension only plot histogram
    if X.shape[1] == 1:
        plot_histogram(X[:, 0], labels, density, n_bins, show_legend)
    else:
        # Get unique labels and unique true labels
        if labels is not None:
            unique_labels = np.unique(labels)
            # Manage colormap
            cmap = cm.get_cmap('viridis', 12)
            norm = Normalize(vmin=unique_labels[0], vmax=unique_labels[-1])
        if true_labels is not None:
            unique_true_labels = np.unique(true_labels)
        # Create subplots
        if equal_axis:
            fig, axes = plt.subplots(nrows=X.shape[1], ncols=X.shape[1], sharey="all", sharex="all")
        else:
            fig, axes = plt.subplots(nrows=X.shape[1], ncols=X.shape[1], sharey="row", sharex="col")
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                ax = axes[i, j]
                if i == j:
                    # Histogram plot
                    if i != 0:
                        ax.yaxis.set_visible(False)
                    if i != X.shape[1] - 1:
                        ax.xaxis.set_visible(False)
                    # Second plot for actual histogram
                    twin_axis = ax.twinx()
                    twin_axis.yaxis.set_visible(False)
                    plot_histogram(X[:, i], labels, density, n_bins, show_legend=False, container=twin_axis,
                                   show_plot=False)
                else:
                    # Scatter plot
                    if true_labels is None:
                        ax.scatter(X[:, j], X[:, i], s=scattersize, c=labels)
                    else:
                        # Change marker for true labels
                        for lab_index, true_lab in enumerate(unique_true_labels):
                            marker = MARKERS[lab_index % len(MARKERS)]
                            ax.scatter(X[true_labels == true_lab, j], X[true_labels == true_lab, i], s=scattersize,
                                       c=labels if labels is None else labels[true_labels == true_lab], marker=marker,
                                       vmin=unique_labels[0], vmax=unique_labels[-1])
                    if centers is not None:
                        ax.scatter(centers[:, j], centers[:, i], s=scattersize * 1.5, color="red", marker="s")
                        for cen_id in range(len(centers)):
                            ax.text(centers[cen_id, j], centers[cen_id, i], str(cen_id), weight="bold")
        if show_legend and labels is not None:
            _add_legend(fig, unique_labels, cmap, norm)
        if show_plot:
            plt.show()
        return axes


def _add_legend(fig, unique_labels, cmap, norm):
    patchlist = [mpatches.Patch(color=cmap(norm(lab)), label=lab) for lab in unique_labels]
    fig.legend(handles=patchlist, loc="center right")
