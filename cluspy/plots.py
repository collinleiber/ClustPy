import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def transformation_plot(X, labels=None, centers=None, plot_3d=False, transformation_class=PCA, figsize=(15, 15),
                        scattersize=10):
    # 2d or 3d plot?
    if plot_3d:
        trans_dimensions = min(X.shape[1], 3)
    else:
        trans_dimensions = min(X.shape[1], 2)
    # Check if transformation dimensionality is smaller than number of features
    if X.shape[1] > trans_dimensions:
        trans = transformation_class(n_components=trans_dimensions)
        X = trans.fit_transform(X)
        if centers is not None:
            centers = trans.transform(centers)
    if trans_dimensions == 2:
        # 2D Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=scattersize)
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], s=scattersize*1.5, c="red", marker="s")
            for j in range(len(centers)):
                ax.text(centers[j, 0], centers[j, 1], str(j), weight="bold")
    else:
        # 3D Plot
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)  # fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], zs=X[:, 2], zdir='z', s=scattersize, c=labels)
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], zs=centers[:, 2], zdir='z', s=scattersize*1.5, c="red", marker="s")
            for j in range(len(centers)):
                ax.text(centers[j, 0], centers[j, 1], centers[j, 2], str(j), weight="bold")
    plt.show()
