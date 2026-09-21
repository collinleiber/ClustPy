from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class ZNormalizer(TransformerMixin, BaseEstimator):
    """
    Normalize a data set by calculating (data - mean) / std.
    In general, two strategies are sensible to normalize a data set.
    Either use all features simultaneously for the normalization or normalize each feature separately.
    In the case of image data, a feature-wise transformation usually corresponds to a channel-wise transformation.
    If this normalizer should be applied to RGB image data, the color channels should be in the first dimension, known as CHW representation.

    Parameters
    ----------
    feature_or_channel_wise : bool
        Specifies if all data should be used for the normalization or if a feature-/channel-wise normalization should be applied (default: False)

    Attributes
    ----------
    shape : list
        Shape of the data set with which this normalizer has been fitted
    mean : np.ndarray or int
        Mean value(s) of the data set
    std : np.ndarray or int
        Standard deviation value(s) of the data set
    """

    def __init__(self, feature_or_channel_wise: bool = False):
        self.feature_or_channel_wise = feature_or_channel_wise

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'ZNormalizer':
        """
        Compute the mean and std values regarding the input data set.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray | None
            the labels (can be ignored)

        Returns
        -------
        self : ZNormalizer
            this instance of the ZNormalizer
        """
        self.shape = list(X.shape)
        self.shape[0] = -1
        if not self.feature_or_channel_wise or (X.ndim > 2 and 3 not in self.shape):
            # In case of not feature_or_channel_wise or grayscale images (2d or 3d)
            self.std = np.std(X)
            self.mean = np.mean(X)
        elif self.feature_or_channel_wise and (X.ndim == 2 or (X.ndim in [4, 5] and X.shape[1] == 3)):
            # In case of tabular data or RGB 2D or 3D images
            self.std = np.array([np.std(X[:, j]) for j in range(self.shape[1])])
            self.mean = np.array([np.mean(X[:, j]) for j in range(self.shape[1])])
        else:
            raise Exception(
                "Your combination of feature_or_channel_wise={0} and X.ndim={1} is not working for the transformation".format(
                    self.feature_or_channel_wise, X.ndim))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the given data set using the fitted mean and std values.

        Parameters
        ----------
        X : np.ndarray
            the given data set

        Returns
        -------
        X_out : np.ndarray
            The transformed data set
        """
        assert list(X.shape)[1:] == self.shape[
                                    1:], "The shape of the input data does not match the fitted transformation. Shape must be {0}".format(
            self.shape)
        X_out = X.astype(float)
        if not self.feature_or_channel_wise or (X.ndim > 2 and 3 not in self.shape):
            # In case of not feature_or_channel_wise or grayscale images if feature_or_channel_wise
            X_out = (X_out - self.mean) / self.std
        elif self.feature_or_channel_wise and X.ndim in [2, 4, 5]:
            # In case of tabular data or RGB 2D or 3D images
            for j in range(self.shape[1]):
                X_out[:, j] = (X_out[:, j] - self.mean[j]) / self.std[j]
        else:
            raise Exception(
                "Your combination of feature_or_channel_wise={0} and X.ndim={1} is not working for the transformation".format(
                    self.feature_or_channel_wise,
                    X.ndim))
        return X_out

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Invert the transformation by applying (data * std) + mean.

        Parameters
        ----------
        X : np.ndarray
            the given data set

        Returns
        -------
        X_out : np.ndarray
            The transformed data set
        """
        assert list(X.shape)[1:] == self.shape[
                                    1:], "The shape of the input data does not match the fitted transformation. Shape must be {0}".format(
            self.shape)
        X_out = X.astype(float)
        if not self.feature_or_channel_wise or (X.ndim > 2 and 3 not in self.shape):
            # In case of not feature_or_channel_wise or grayscale images if feature_or_channel_wise
            X_out = X_out * self.std + self.mean
        elif self.feature_or_channel_wise and X.ndim in [2, 4, 5]:
            # In case of tabular data or RGB 2D or 3D images
            for j in range(self.shape[1]):
                X_out[:, j] = X_out[:, j] * self.std[j] + self.mean[j]
        else:
            raise Exception(
                "Your combination of feature_or_channel_wise={0} and X.ndim={1} is not working for the transformation".format(
                    self.feature_or_channel_wise,
                    X.ndim))
        return X_out


def z_normalization(X: np.ndarray, feature_or_channel_wise: bool = False) -> np.ndarray:
    """
    Wrapper for the ZNormalizer.
    It automatically executes: X_transform = ZNormalizer(feature_or_channel_wise).fit_transform(X)

    Parameters
    ----------
    X : np.ndarray
        the given data set
    feature_or_channel_wise : bool
        Specifies if all data should be used for the normalization or if a feature-/channel-wise normalization should be applied (default: False)

    Returns
    -------
    X_transform : np.ndarray
        The transformed data set
    """
    znorm = ZNormalizer(feature_or_channel_wise)
    X_transform = znorm.fit_transform(X)
    return X_transform


def bm25(X: np.ndarray, k: float=1.5, b: float = 0.75) -> np.ndarray:
    """
    Perform a BM25 transformation.
    Takes a matrix containing word counts and changes its representation by considering the number of documents containing a word and the length of the documents.
    Similiar to TF-IDF.

    Parameters
    ----------
    X : np.ndarray
        the given data set
    k : float
        the k parameter of BM25. Usually within [1.2, 2.0] (default: 1.5)
    b : float
        the b parameter of BM25. Has to be within [0, 1] (default: 0.75)

    Returns
    -------
    X_transform : np.ndarray
        The transformed data set

    References
    ----------
    Robertson, Stephen, and Hugo Zaragoza. "The probabilistic relevance framework: BM25 and beyond."
    Foundations and trends in information retrieval 4.1-2 (2009): 1-174.
    """
    assert np.all(X >= 0), "X must be non-negative (counts)"
    assert k > 0, "k must be > 0"
    assert 0.0 <= b <= 1.0, "b must be in [0, 1]"
    X = np.asarray(X, dtype=float)
    n_documents_with_q = (X > 0).sum(0)
    N = X.shape[0]
    idf = np.log((N - n_documents_with_q + 0.5) / (n_documents_with_q + 0.5) + 1)
    length_D = X.sum(1)
    avgdl = length_D.mean()
    if avgdl == 0:
        return np.zeros_like(X)
    bm25 = X * (k + 1) / (X + k * (1 - b + b * length_D.reshape((-1, 1)) / avgdl))
    X_transform = bm25 * idf.reshape((1, -1))
    return X_transform
