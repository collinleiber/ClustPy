import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math


def plot_with_transformation(X, labels=None, centers=None, plot_dimensionality=2, transformation_class=PCA,
                             scattersize=10):
    assert plot_dimensionality <= X.shape[1], "Plot dimensionality can not be larger than the number of features."
    assert plot_dimensionality > 0 and plot_dimensionality < 4, "Plot dimensionality must be > 0 and < 4."
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
            plt.scatter(centers[:, 0], yc, s=300, c="red", marker="x")
            # plot one center text above line and next below ...
            centers_order = np.argsort(centers[:, 0])
            centers_order = np.argsort(centers_order)
            for j in range(len(centers)):
                yt = 1.0005 if centers_order[j] % 2 == 0 else 0.9994
                plt.text(centers[j, 0], yt, str(j), weight="bold")
    elif plot_dimensionality == 2:
        # 2d Plot
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=scattersize)
        if centers is not None:
            plt.scatter(centers[:, 0], centers[:, 1], s=scattersize * 1.5, c="red", marker="s")
            for j in range(len(centers)):
                plt.text(centers[j, 0], centers[j, 1], str(j), weight="bold")
        plt.axis("equal")
    elif plot_dimensionality == 3:
        # 3d Plot
        fig = plt.figure()
        ax = Axes3D(fig)  # fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], zs=X[:, 2], zdir='z', s=scattersize, c=labels)
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], zs=centers[:, 2], zdir='z', s=scattersize * 1.5, c="red",
                       marker="s")
            for j in range(len(centers)):
                ax.text(centers[j, 0], centers[j, 1], centers[j, 2], str(j), weight="bold")
    else:
        pass
        # Nd Plot -> add scatter matrix plot
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
