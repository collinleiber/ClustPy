import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Colormap
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats
import matplotlib.patches as mpatches
from sklearn.base import TransformerMixin

"""
Constants
"""
# Circle, Square, Diamond, Plus, X, Triangle down, Star, Pentagon, Triangle Up, Triangle left, Triangle right, Hexagon
_MARKERS = ("o", "s", "D", "P", "X", "v", "*", "p", "^", ">", "<", "h")
_MIN_OBJECTS_FOR_DENS_PLOT = 3


def plot_with_transformation(X: np.ndarray, labels: np.ndarray = None, centers: np.ndarray = None,
                             true_labels: np.ndarray = None, plot_dimensionality: int = 2,
                             transformation_class: TransformerMixin = PCA, show_legend: bool = True,
                             scattersize: float = 10, equal_axis: bool = False, show_plot: bool = True) -> None:
    """
    In Data Science, it is common to work with high-dimensional data.
    These cannot be visualized without further ado.
    Therefore, a dimensionality reduction technique is often applied before a plot is created.
    Examples for such techniques are PCA, ICA, t-SNE, UMAP, ...
    Note that the chosen technique must work with a 'fit_transform' method.

    This method automatically executes the aforementioned pipline:
    first it reduces the dimensionality, then it creates a plot adjusted to the number of features.
    Up to three dimensions are visualized with the help of scatter plats. Then a scatter matrix plot is used.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    labels : np.ndarray
        The cluster labels. Specifies the color of the plotted objects. Can be None (default: None)
    centers : np.ndarray
        The cluster centers. Will be plotted as red dots labeled by the corresponding cluster id. Can be None (default: None)
    true_labels : np.ndarray
        The ground truth labels. Specifies the symbol of the plotted objects. Can be None (default: None)
    plot_dimensionality : int
        The dimensionality of the feature space after the dimensionality reduction technique has been applied (default: 2)
    transformation_class : TransformerMixin
        The transformation class / dimensionality reduction technique (default: sklearn.decomposition.PCA)
    show_legend : bool
        Defines whether a legend should be shown (default: True)
    scattersize : float
        The size of the scatters (default: 10)
    equal_axis : bool
        Defines whether the axes are to be scaled to the same value range (default: False)
    show_plot : bool
        Defines whether the plot should directly be plotted (default: True)
    """
    assert plot_dimensionality > 0, "Plot dimensionality must be > 0"
    if X.ndim == 1:
        plot_dimensionality = 1
    elif plot_dimensionality > X.shape[1]:
        print(
            "[WARNING] plot_dimensionality ({0}) is higher than the dimensionaliyty of the input dataset ({1}). "
            "plot_dimensionality will therefore be set to {1}.".format(
                plot_dimensionality, X.shape[1]))
        plot_dimensionality = X.shape[1]
    # Check if transformation dimensionality is smaller than number of features
    elif plot_dimensionality < X.shape[1]:
        # Transfrom data
        trans = transformation_class(n_components=plot_dimensionality)
        X = trans.fit_transform(X)
        if centers is not None:
            centers = trans.transform(centers)
    # Create plot
    if plot_dimensionality == 1:
        # 1d Plot
        plot_1d_data(X, labels=labels, centers=centers, true_labels=true_labels, show_legend=show_legend,
                     show_plot=False)
    elif plot_dimensionality == 2:
        # 2d Plot
        plot_2d_data(X, labels=labels, centers=centers, true_labels=true_labels, show_legend=show_legend,
                     scattersize=scattersize, equal_axis=equal_axis, show_plot=False)
    elif plot_dimensionality == 3:
        # 3d Plot
        plot_3d_data(X, labels=labels, centers=centers, true_labels=true_labels, show_legend=show_legend,
                     scattersize=scattersize, show_plot=False)
    else:
        # More than 3 features
        plot_scatter_matrix(X, labels=labels, centers=centers, true_labels=true_labels, scattersize=scattersize,
                            show_legend=show_legend, equal_axis=equal_axis, max_dimensions=plot_dimensionality,
                            show_plot=False)
    if show_plot:
        plt.show()


def plot_1d_data(X: np.ndarray, labels: np.ndarray = None, centers: np.ndarray = None, true_labels: np.ndarray = None,
                 show_legend: bool = True, show_plot: bool = True) -> None:
    """
    Plot a one-dimensional data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    labels : np.ndarray
        The cluster labels. Specifies the color of the plotted objects. Can be None (default: None)
    centers : np.ndarray
        The cluster centers. Will be plotted as red dots labeled by the corresponding cluster id. Can be None (default: None)
    true_labels : np.ndarray
        The ground truth labels. Specifies the symbol of the plotted objects. Can be None (default: None)
    show_legend : bool
        Defines whether a legend should be shown (default: True)
    show_plot : bool
        Defines whether the plot should directly be plotted (default: True)
    """
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
    if show_legend and labels is not None:
        unique_labels, cmap, norm = _get_cmap_and_norm(labels)
        _add_legend(plt, unique_labels, cmap, norm)
    if show_plot:
        plt.show()


def plot_2d_data(X: np.ndarray, labels: np.ndarray = None, centers: np.ndarray = None, true_labels: np.ndarray = None,
                 cluster_ids_font_size: float = None, centers_ids_font_size: float = 10, show_legend: bool = True,
                 title: str = None, scattersize: float = 10, centers_scattersize: float = 15, equal_axis: bool = False,
                 container: plt.Axes = plt, show_plot: bool = True) -> None:
    """
    Plot a two-dimensional data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    labels : np.ndarray
        The cluster labels. Specifies the color of the plotted objects. Can be None (default: None)
    centers : np.ndarray
        The cluster centers. Will be plotted as red dots labeled by the corresponding cluster id. Can be None (default: None)
    true_labels : np.ndarray
        The ground truth labels. Specifies the symbol of the plotted objects. Can be None (default: None)
    cluster_ids_font_size : float
        The font size of the id of a predicted cluster, which is shown as text in the center of that cluster.
        Can be None if no id should be shown (default: None)
    centers_ids_font_size: float
        The font size of the id that is shown next to the red marker of a cluster center. Only relevant if centers is not None.
        Can be None if no id should be shown (default: 10)
    show_legend : bool
        Defines whether a legend should be shown (default: True)
    title : str
        Title of the plot (default: None)
    scattersize : float
        The size of the scatters (default: 10)
    centers_scattersize : float
        The size of the red scatters of the cluster centers (default: 15)
    equal_axis : bool
        Defines whether the axes are to be scaled to the same value range (default: False)
    container : plt.Axes
        The container to which the scatter plot is added.
        If another container is defined, show_plot should usually be False (default: matplotlib.pyplot)
    show_plot : bool
        Defines whether the plot should directly be plotted (default: True)
    """
    assert X.ndim == 2 and X.shape[1] == 2, "Data must be 2-dimensional"
    if true_labels is None:
        container.scatter(X[:, 0], X[:, 1], c=labels, s=scattersize)
    else:
        unique_true_labels = np.unique(true_labels)
        # Change marker for true labels
        for lab_index, true_lab in enumerate(unique_true_labels):
            marker = _MARKERS[lab_index % len(_MARKERS)]
            container.scatter(X[true_labels == true_lab, 0], X[true_labels == true_lab, 1], s=scattersize,
                              c=labels if labels is None else labels[true_labels == true_lab], marker=marker,
                              vmin=np.min(labels), vmax=np.max(labels))
    if cluster_ids_font_size is not None:
        unique_labels = np.unique(labels)
        mean_positions = [np.mean(X[labels == pred_lab], axis=0) for pred_lab in unique_labels]
        for i, mp in enumerate(mean_positions):
            plt.text(mp[0], mp[1], unique_labels[i], fontsize=cluster_ids_font_size)
    if centers is not None:
        container.scatter(centers[:, 0], centers[:, 1], s=centers_scattersize, color="red", marker="s")
        if centers_ids_font_size is not None:
            for j in range(len(centers)):
                container.text(centers[j, 0], centers[j, 1], str(j), weight="bold", fontsize=centers_ids_font_size)
    if equal_axis:
        container.axis("equal")
    if show_legend and labels is not None:
        unique_labels, cmap, norm = _get_cmap_and_norm(labels)
        _add_legend(container, unique_labels, cmap, norm)
    if title is not None:
        plt.title(title)
    if show_plot:
        container.show()


def plot_3d_data(X: np.ndarray, labels: np.ndarray = None, centers: np.ndarray = None, true_labels: np.ndarray = None,
                 show_legend: bool = True, scattersize: float = 10, show_plot: bool = True) -> None:
    """
    Plot a three-dimensional data set.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    labels : np.ndarray
        The cluster labels. Specifies the color of the plotted objects. Can be None (default: None)
    centers : np.ndarray
        The cluster centers. Will be plotted as red dots labeled by the corresponding cluster id. Can be None (default: None)
    true_labels : np.ndarray
        The ground truth labels. Specifies the symbol of the plotted objects. Can be None (default: None)
    show_legend : bool
        Defines whether a legend should be shown (default: True)
    scattersize : float
        The size of the scatters (default: 10)
    show_plot : bool
        Defines whether the plot should directly be plotted (default: True)
    """
    assert X.ndim == 2 or X.shape[1] == 3, "Data must be 3-dimensional"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Axes3D(fig)
    if true_labels is None:
        ax.scatter(X[:, 0], X[:, 1], zs=X[:, 2], zdir='z', s=scattersize, c=labels, alpha=0.8)
    else:
        unique_true_labels = np.unique(true_labels)
        # Change marker for true labels
        for lab_index, true_lab in enumerate(unique_true_labels):
            marker = _MARKERS[lab_index % len(_MARKERS)]
            ax.scatter(X[true_labels == true_lab, 0], X[true_labels == true_lab, 1],
                       zs=X[true_labels == true_lab, 2], zdir='z', s=scattersize,
                       c=labels if labels is None else labels[true_labels == true_lab],
                       marker=marker, vmin=np.min(labels), vmax=np.max(labels), alpha=0.8)
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], zs=centers[:, 2], zdir='z', s=scattersize * 1.5, color="red",
                   marker="s")
        for j in range(len(centers)):
            ax.text(centers[j, 0], centers[j, 1], centers[j, 2], str(j), weight="bold")
    if show_legend and labels is not None:
        unique_labels, cmap, norm = _get_cmap_and_norm(labels)
        _add_legend(fig, unique_labels, cmap, norm)
    if show_plot:
        plt.show()


def plot_image(img_data: np.ndarray, black_and_white: bool = False, image_shape: tuple = None,
               is_color_channel_last: bool = False, max_value: float = None, min_value: float = None,
               show_plot: bool = True) -> None:
    """
    Plot an image.
    Color image should occur in the HWC representation (height, width, color channels) if is_color_channel_last is True and in the CHW if is_color_channel_last is False.

    Parameters
    ----------
    img_data : np.ndarray
        The image data
    black_and_white : bool
        Specifies whether the image should be plotted in grayscale colors. Only relevant for images without color channels (default: False)
    image_shape : tuple
        (height, width) for grayscale images or HWC (height, width, color channels) / CHW for color images (default: None)
    is_color_channel_last : bool
        if true, the color channels should be in the last dimension, known as HWC representation. Alternatively the color channel can be at the first position, known as CHW representation.
        Only relevant for color images (default: False)
    max_value : float
        maximum pixel value, used for min-max normalization. Is often 255, if None the maximum value in the data set will be used (default: None)
    min_value : float
        maximum pixel value, used for min-max normalization. Is often 0, if None the minimum value in the data set will be used (default: 255)
    show_plot : bool
        Defines whether the plot should directly be plotted (default: True)

    Examples
    ----------
    >>> from clustpy.data import load_nrletters, load_optdigits
    >>> X = load_nrletters().data
    >>> plot_image(X[0], False, (9, 7, 3), True, 255, 0, show_plot=True)
    >>> X = load_optdigits().data
    >>> plot_image(X[0], True, (8, 8), None, 255, 0, show_plot=True)
    """
    assert img_data.ndim <= 3, "Image data can not have more than 3 dimensions."
    # Data range must match float between [0..1] or int between [0..255] -> use min-max transform
    if max_value is None:
        max_value = np.max(img_data)
    if min_value is None:
        min_value = np.min(img_data)
    # Scale image to [0, 1]
    img_data = (img_data - min_value) / (max_value - min_value)
    # Reshape array data
    if img_data.ndim == 1:
        img_data = img_data.reshape(image_shape)
    if img_data.ndim == 3 and not is_color_channel_last:
        # Reshape image to HWC representation
        img_data = np.transpose(img_data, (1, 2, 0))
    # Plot original image or a black-and-white version
    if black_and_white:
        plt.imshow(img_data, cmap="Greys")
    else:
        plt.imshow(img_data)
    plt.axis('off')
    if show_plot:
        plt.show()


def plot_histogram(X: np.ndarray, labels: np.ndarray = None, density: bool = True, n_bins: int = 100,
                   show_legend: bool = True, container: plt.Axes = plt, show_plot: bool = True) -> None:
    """
    Plot a histogram.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    labels : np.ndarray
        The cluster labels. Specifies the color of the plotted objects. Can be None (default: None)
    density : bool
        Defines whether a kernel density should be added to the histogram (default: True)
    n_bins : int
        Number of bins (default: 100)
    show_legend : bool
        Defines whether the legend of the histogram should be shown (default: True)
    container : plt.Axes
        The container to which the histogram is added.
        If another container is defined, show_plot should usually be False (default: matplotlib.pyplot)
    show_plot : bool
        Defines whether the plot should directly be plotted (default: True)
    """
    assert X.ndim == 1, "Data must be 1-dimensional"
    # Plot histogram
    if labels is not None:
        unique_labels, cmap, norm = _get_cmap_and_norm(labels)
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


def plot_scatter_matrix(X: np.ndarray, labels: np.ndarray = None, centers: np.ndarray = None,
                        true_labels: np.ndarray = None, density: bool = True, n_bins: int = 100,
                        show_legend: bool = True, scattersize: float = 10, equal_axis: bool = False,
                        max_dimensions: int = 10, show_plot: bool = True) -> plt.Axes:
    """
    Create a scatter matrix plot.
    Visualizes a 2d scatter plot for each combination of features.
    The center axis shows a histogram of each single feature.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    labels : np.ndarray
        The cluster labels. Specifies the color of the plotted objects. Can be None (default: None)
    centers : np.ndarray
        The cluster centers. Will be plotted as red dots labeled by the corresponding cluster id. Can be None (default: None)
    true_labels : np.ndarray
        The ground truth labels. Specifies the symbol of the plotted objects. Can be None (default: None)
    density : bool
        Defines whether a kernel density should be added to the histogram (default: True)
    n_bins : int
        Number of bins used for the histogram (default: 100)
    show_legend : bool
        Defines whether a legend should be shown (default: True)
    scattersize : float
        The size of the scatters (default: 10)
    equal_axis : bool
        Defines whether the axes are to be scaled to the same value range (default: False)
    max_dimensions : int
        Maximum Number of dimensions that should be plotted.
        This value is intended to prevent the creation of overly complex plots that are very confusing and take a long time to create (default: 10)
    show_plot : bool
        Defines whether the plot should directly be plotted (default: True)

    Returns
    -------
    axes : plt.Axes
        None if show_plot is True, otherwise the used matplotlib axes
    """
    if X.shape[1] > max_dimensions:
        print(
            "[WARNING] Dimensionality of the dataset is larger than 10. Creation of scatter matrix plot will be aborted.")
    # For single dimension only plot histogram
    if X.shape[1] == 1:
        plot_histogram(X[:, 0], labels, density, n_bins, show_legend, show_plot=show_plot)
        if not show_plot:
            return plt.gca()
    else:
        # Get unique labels and unique true labels
        if labels is not None:
            unique_labels, cmap, norm = _get_cmap_and_norm(labels)
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
                    # Second plot for actual histogram (use container)
                    twin_axis = ax.twinx()
                    twin_axis.yaxis.set_visible(False)
                    plot_histogram(X[:, i], labels, density, n_bins, show_legend=False, container=twin_axis,
                                   show_plot=False)
                else:
                    # Scatter plot (use container)
                    local_centers = None if centers is None else centers[:, [j, i]]
                    plot_2d_data(X[:, [j, i]], labels, local_centers, true_labels, show_legend=False,
                                 scattersize=scattersize,
                                 equal_axis=False, container=ax, show_plot=False)
        if show_legend and labels is not None:
            _add_legend(fig, unique_labels, cmap, norm)
        if show_plot:
            plt.show()
        else:
            return axes


def _add_legend(container: plt.Axes, unique_labels: np.ndarray, cmap: Colormap, norm: Normalize) -> None:
    """
    Helper function to add a legend to the histogram.

    Parameters
    ----------
    container : plt.Axes
        The container to which the legend is added.
    unique_labels : np.ndarray
        The unique labels that should be displayed in the legend
    cmap : Colormap
        the colormap
    norm : Normalize
        The Normalize object to pick the correct color
    """
    patchlist = [mpatches.Patch(color=cmap(norm(lab)), label=lab) for lab in unique_labels]
    container.legend(handles=patchlist, loc="center right")


def _get_cmap_and_norm(labels: np.ndarray, min_max: tuple = None) -> (np.ndarray, Colormap, Normalize):
    """
    Helper function to get colormap and Normalization object.

    Parameters
    ----------
    labels : np.ndarray
        The cluster labels
    min_max : tuple
        Tuple containing the minimum and maximum cluster label for coloring the plot (default: None)

    Returns
    -------
    tuple : (np.ndarray, Colormap, Normalize)
        The unique labels ids,
        The colormap,
        The Normalize object to pick the correct color
    """
    unique_labels = np.unique(labels)
    if min_max is None:
        min_max = (unique_labels[0], unique_labels[-1])
    assert min_max[0] <= min_max[1], "First value in min_max must be smaller or equal to second value"
    # Manage colormap
    cmap = matplotlib.colormaps['viridis']
    norm = Normalize(vmin=min_max[0], vmax=min_max[1])
    return unique_labels, cmap, norm
