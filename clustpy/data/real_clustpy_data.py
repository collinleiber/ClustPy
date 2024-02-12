import numpy as np
import os
from sklearn.datasets._base import Bunch
from clustpy.data._utils import unflatten_images


def _load_nr_data(file_name: str, n_labels: int) -> (np.ndarray, np.ndarray):
    """
    Helper function to load a non-redundant data set from ClustPys internal data sets directory.
    The first n_labels columns will be specified as labels.

    Parameters
    ----------
    file_name: str
        Name of the data set
    n_labels: int
        Number of label sets

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array, the labels numpy array
    """
    path = os.path.dirname(__file__) + "/datasets/" + file_name
    dataset = np.genfromtxt(path, delimiter=",")
    data = dataset[:, n_labels:]
    labels = dataset[:, :n_labels]
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    return data, labels


"""
Actual datasets
"""


def load_aloi_small(return_X_y: bool = False) -> Bunch:
    """
    Load a subset of the Amsterdam Library of Object Image (ALOI) consisting of 288 images of the objects red ball,
    red cylinder, green ball and green cylinder. The two label sets are cylinder/ball and red/green.
    N=288, d=611, k=[2,2].

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (288 x 611), the labels numpy array (288 x 2)

    References
    -------
    https://aloi.science.uva.nl/

    and

    Ye, Wei, et al. "Generalized independent subspace clustering." 2016 IEEE 16th International Conference on Data
    Mining (ICDM). IEEE, 2016.
    """
    data, labels = _load_nr_data("aloi_small.data", 2)
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="ALOI_small", data=data, target=labels)


def load_fruit(return_X_y: bool = False) -> Bunch:
    """
    Load the fruits data set. It consists of 105 preprocessed images of apples, bananas and grapes in red, green and yellow.
    N=105, d=6, k=[3,3].

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (105 x 6), the labels numpy array (105 x 2)

    References
    -------
    Hu, Juhua, et al. "Finding multiple stable clusterings." Knowledge and Information Systems 51.3 (2017): 991-1021.
    """
    data, labels = _load_nr_data("fruit.data", 2)
    if return_X_y:
        return data, labels
    else:
        return Bunch(dataset_name="FRUIT", data=data, target=labels)


def load_nrletters(return_X_y: bool = False) -> Bunch:
    """
    Load the NRLetters data set. It consists of 10000 9x7 images of the letters A, B, C, X, Y and Z in pink, cyan and
    yellow. Additionally, each image highlights one corner in color.
    N=10000, d=189, k=[6,3,4].

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (10000 x 189), the labels numpy array (10000 x 3)

    References
    -------
    Leiber, Collin, et al. "Automatic Parameter Selection for Non-Redundant Clustering." Proceedings of the 2022 SIAM
    International Conference on Data Mining (SDM). Society for Industrial and Applied Mathematics, 2022.
    """
    data, labels = _load_nr_data("nrLetters.data", 3)
    if return_X_y:
        return data, labels
    else:
        data_image = unflatten_images(data, (9, 7, 3))
        return Bunch(dataset_name="NrLetters", data=data, target=labels, images=data_image,
                     image_format="CHW")


def load_stickfigures(return_X_y: bool = False) -> Bunch:
    """
    Load the Dancing Stick Figures data set. It consists of 900 20x20 grayscale images of stick figures in different poses.
    The poses can be divided into three upp-body and three lower-body motions.
    N=900, d=400, k=[3,3].

    Parameters
    ----------
    return_X_y : bool
        If True, returns (data, target) instead of a Bunch object. See below for more information about the data and target object (default: False)

    Returns
    -------
    bunch : Bunch
        A Bunch object containing the data in the 'data' attribute and the labels in the 'target' attribute.
        Furthermore, the original images are contained in the 'images' attribute.
        Note that the data within 'data' is in HWC format and within 'images' in the CHW format.
        Alternatively, if return_X_y is True two arrays will be returned:
        the data numpy array (900 x 400), labels: the labels numpy array (900 x 2)

    References
    -------
    Günnemann, Stephan, et al. "Smvc: semi-supervised multi-view clustering in subspace projections." Proceedings of
    the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014.
    """
    data, labels = _load_nr_data("stickfigures.data", 2)
    if return_X_y:
        return data, labels
    else:
        data_image = unflatten_images(data, (20, 20))
        return Bunch(dataset_name="Stickfigures", data=data, target=labels, images=data_image,
                     image_format="HW")
