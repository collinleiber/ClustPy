import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from scipy import stats
import matplotlib.patches as mpatches

# Circle, Square, Diamond, Plus, X, Triangle down, Star, Pentagon, Triangle Up, Triangle left, Triangle right, Hexagon
MARKERS = ["o", "s", "D", "P", "X", "v", "*", "p", "^", ">", "<", "h"]

_MIN_OBJECTS_FOR_DENS_PLOT = 3

def plot_with_transformation(X, labels=None, centers=None, true_labels=None, plot_dimensionality=2,
                             transformation_class=PCA,
                             scattersize=10, equal_axis=False):
    assert plot_dimensionality <= X.shape[1], "Plot dimensionality can not be larger than the number of features."
    assert plot_dimensionality > 0, "Plot dimensionality must be > 0"
    if true_labels is not None:
        unique_true_labels = np.unique(true_labels)
    # Check if transformation dimensionality is smaller than number of features
    trans = transformation_class(n_components=plot_dimensionality)
    X = trans.fit_transform(X)
    if centers is not None:
        centers = trans.transform(centers)
    if plot_dimensionality == 1:
        # 1d Plot
        # fig, ax = plt.subplots(figsize=figsize)
        plt.hlines(1, np.min(X), np.max(X))  # Draw a horizontal line
        y = np.ones(len(X))
        plt.scatter(X, y, marker='|', s=500, c=labels)  # Plot a line at each location specified in X
        if centers is not None:
            yc = np.ones(len(centers))
            plt.scatter(centers[:, 0], yc, s=300, color="red", marker="x")
            # plot one center text above line and next below ...
            centers_order = np.argsort(centers[:, 0])
            centers_order = np.argsort(centers_order)
            for j in range(len(centers)):
                yt = 1.0005 if centers_order[j] % 2 == 0 else 0.9994
                plt.text(centers[j, 0], yt, str(j), weight="bold")
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
                            equal_axis=equal_axis)
    if plot_dimensionality <= 3:
        plt.show()


def plot_image(img_data, black_and_white=True, image_shape=None):
    assert img_data.ndim <= 2, "Image data can not be larger than 2."
    if img_data.ndim == 1:
        if image_shape is None:
            sqrt_of_data = int(math.sqrt(len(img_data)))
            assert len(img_data) == sqrt_of_data ** 2, "Image shape must be specified or image must be square."
            image_shape = (sqrt_of_data, sqrt_of_data)
        img_data = img_data.reshape(image_shape)
    if black_and_white:
        plt.imshow(img_data, cmap="Greys")
    else:
        plt.imshow(img_data)
    plt.axis('off')
    plt.show()


def plot_histogram(X, labels=None, density=True, n_bins=100, show_legend=True):
    if labels is not None:
        unique_labels = np.unique(labels)
        # Manage colormap
        cmap = cm.get_cmap('viridis', 12)
        norm = Normalize(vmin=unique_labels[0], vmax=unique_labels[-1])
    # Plot histogram
    if labels is not None:
        for lab in unique_labels:
            hist_color = cmap(norm(lab))
            plt.hist(X[labels == lab], alpha=0.5, bins=n_bins, color=hist_color, range=(np.min(X), np.max(X)))
    else:
        plt.hist(X, alpha=0.5, bins=n_bins)
    # Plot densities
    if density:
        twin_axis = plt.gca().twinx()
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
            plt.plot(steps, kde(steps))
    if show_legend and labels is not None:
        _add_legend(plt, unique_labels, cmap, norm)
    plt.show()


def plot_scatter_matrix(X, labels=None, centers=None, true_labels=None, density=True, n_bins=100, show_legend=True,
                        scattersize=10, equal_axis=False):
    if X.shape[1] > 10:
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
                    if labels is not None:
                        for lab in unique_labels:
                            hist_color = cmap(norm(lab))
                            twin_axis.hist(X[labels == lab, i], alpha=0.5, bins=n_bins,
                                           color=hist_color, range=(np.min(X[:, i]), np.max(X[:, i])))
                    else:
                        twin_axis.hist(X[:, i], alpha=0.5, bins=n_bins)
                    # Plot densities
                    if density:
                        twin_axis2 = ax.twinx()
                        twin_axis2.yaxis.set_visible(False)
                        if labels is not None:
                            for lab in unique_labels:
                                den_objects = X[labels == lab]
                                if den_objects.shape[0] >= _MIN_OBJECTS_FOR_DENS_PLOT:
                                    hist_color = cmap(norm(lab))
                                    kde = stats.gaussian_kde(den_objects[:, i])
                                    steps = np.linspace(np.min(den_objects[:, i]), np.max(den_objects[:, i]), 1000)
                                    twin_axis2.plot(steps, kde(steps), color=hist_color)
                        elif X[:, i].shape[0] >= _MIN_OBJECTS_FOR_DENS_PLOT:
                            kde = stats.gaussian_kde(X[:, i])
                            steps = np.linspace(np.min(X[:, i]), np.max(X[:, i]), 1000)
                            twin_axis2.plot(steps, kde(steps))
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
        plt.show()


def _add_legend(fig, unique_labels, cmap, norm):
    patchlist = [mpatches.Patch(color=cmap(norm(lab)), label=lab) for lab in unique_labels]
    fig.legend(handles=patchlist, loc="center right")