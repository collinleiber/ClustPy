"""
Schelling, Benjamin, et al. "Utilizing Structure-rich Features to improve Clustering." (2020).
The European Conference on Machine Learning and Principles and Practice of Knowledge Discovery
in Databases 2020

@authors: Benjamin Schelling and Sam Maurus (original R implementation), Collin Leiber (Python implementation)
"""

from cluspy.utils import dip_test, dip_gradient
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClusterMixin


def _dip_ext(X, n_components, do_dip_scaling, step_size, momentum, dip_threshold, n_starting_points,
             check_duplicates):
    assert n_components is None or n_components < X.shape[
        1], "n_components must be None or smaller than the dimensionality of the dataset."
    subspace = np.zeros((X.shape[0], 0))
    dip_values = []
    max_dip = 0
    transformed_data = X
    while True:
        dip_value, projection, projected_data = _find_max_dip_by_sgd(transformed_data, step_size, momentum,
                                                                     n_starting_points, check_duplicates)
        if dip_value < max_dip * dip_threshold and n_components is None:
            break
        # Always use the highest dip value
        max_dip = max(dip_value, max_dip)
        dip_values.append(dip_value)
        # Moke projection orthogonal
        # TODO Erweitere Projectionen, um vollständige Dimensionalität zu erhalten -> für fit(X)
        subspace = np.c_[subspace, projected_data]
        if subspace.shape[1] == X.shape[1] or subspace.shape[1] == n_components:
            break
        # Prepare next iteration
        orthogonal_space, _ = np.linalg.qr(projection.reshape(-1, 1), mode="complete")
        transformed_data = np.matmul(transformed_data, orthogonal_space[:, 1:])
    argsorted_dip = np.argsort(dip_values)[::-1]
    dip_values = np.array(dip_values)[argsorted_dip]
    subspace = subspace[:, argsorted_dip]
    if do_dip_scaling:
        subspace = _dip_scaling(subspace, dip_values)
    return subspace, dip_values


def _dip_scaling(X, dip_values):
    X = np.array([dip_values[i] * (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i])) for i in
                  range(X.shape[1])]).T
    return X


def _dip_init(subspace, n_clusters):
    how_many = int(subspace.shape[0] / n_clusters)
    # Get the first initialisation by frequency-binning the primary feature
    centers = np.zeros((n_clusters, 1))
    sorted_primary = np.sort(subspace[:, 0])
    for i in range(n_clusters):
        total = 0
        for j in range(how_many):
            total = total + sorted_primary[i * how_many + j]
        centers[i] = total / how_many
    km = KMeans(n_clusters=n_clusters, init=centers, n_init=1)
    km.fit(subspace[:, 0].reshape(-1, 1))
    if subspace.shape[1] > 1:
        # Add features one by one depending on their dip value
        for i in range(1, subspace.shape[1]):
            centers = np.array([np.mean(subspace[km.labels_ == clus, :i + 1], axis=0) for clus in range(n_clusters)])
            km = KMeans(n_clusters=n_clusters, init=centers, n_init=1)
            km.fit(subspace[:, :i + 1])
    return km.labels_, km.cluster_centers_


def _dip_gradient(X, projection_vector, check_duplicates):
    # projection_vector = projection_vector / np.linalg.norm(projection_vector)  ## Unit vector to start

    ## Project (making a univariate sample)
    projected_data = np.matmul(X, projection_vector)

    sortedIndices = np.argsort(projected_data, kind="stable")
    sorted_projected_data = projected_data[sortedIndices]
    ## Run the dip algorithm, capturing the output which we need for touching-triangle calculations
    dip_value, _, modal_triangle = dip_test(sorted_projected_data, just_dip=False, is_data_sorted=True)
    if modal_triangle is None:
        return [np.zeros(X.shape[1])], dip_value, projected_data
    triangles = [[modal_triangle[0]], [modal_triangle[1]], [modal_triangle[2]]]
    gradients = []
    # Add indices with same value to triangle
    if check_duplicates:
        for j, triangle_point in enumerate(modal_triangle):
            i = 1
            while triangle_point - i > 0 and sorted_projected_data[triangle_point - i] == sorted_projected_data[
                triangle_point]:
                triangles[j].append(triangle_point - i)
                i += 1
            i = 1
            while i + triangle_point < len(sorted_projected_data) and sorted_projected_data[triangle_point + i] == \
                    sorted_projected_data[triangle_point]:
                triangles[j].append(triangle_point + i)
                i += i
    for i1 in triangles[0]:
        for i2 in triangles[1]:
            for i3 in triangles[2]:
                if i1 == i2 or i2 == i3:
                    continue
                # Calculate the partial derivative for all dimensions
                gradient = dip_gradient(X, projected_data, sortedIndices, modal_triangle)
                gradients.append(gradient)
    return gradients, dip_value, projected_data


def _calculate_partial_derivative(projection_vector, dim, i1, i2, i3, beta, gamma, n):
    aBeta = np.sum(beta * projection_vector)  ## This is then equal to (x(i2)-x(i1))
    aGamma = np.sum(gamma * projection_vector)  ## This is then equal to (x(i3)-x(x1))

    eta = i2 - i1 - (((i3 - i1) * (aBeta)) / (aGamma))

    if eta == 0:
        raise Exception("eta zero, which should never be the case")

    betaiGamma = beta[dim] * gamma
    gammaiBeta = gamma[dim] * beta
    firstFraction = (i3 - i1) / n

    secondFraction = ((np.sum((betaiGamma - gammaiBeta) * projection_vector))) / (aGamma ** 2)
    derivative = firstFraction * secondFraction

    if eta > 0:
        return -derivative
    else:
        return derivative


# Find the axes with log(dim) highest dip values and start the search from there
def _find_max_dip_by_sgd(X, step_size, momentum, n_starting_points, check_duplicates):
    axis_dips = [dip_test(X[:, i], just_dip=True, is_data_sorted=False) for i in range(X.shape[1])]
    dips_argsorted = np.argsort(axis_dips)[::-1]
    max_dip = axis_dips[dips_argsorted[0]]
    best_projection = np.zeros(X.shape[1])
    best_projection[dips_argsorted[0]] = 1
    best_projected_data = X[:, dips_argsorted[0]]
    for i in range(n_starting_points):
        start_projection = np.zeros(X.shape[1])
        start_projection[dips_argsorted[i]] = 1
        dip_value, projection, projected_data = _find_max_dip_by_sgd_with_start(X, start_projection, step_size,
                                                                                momentum, check_duplicates)
        if dip_value > max_dip:
            max_dip = dip_value
            best_projection = projection
            best_projected_data = projected_data
    return max_dip, best_projection, best_projected_data


# Gradient Descent to find the projection vector with the maximal dip value
def _find_max_dip_by_sgd_with_start(X, start_projection, step_size, momentum, check_duplicates):
    # Values for GD. Can be changed
    if X.shape[1] == 1:
        dip_value = dip_test(X[:, 0], just_dip=True, is_data_sorted=False)
        return dip_value, np.array([1]), X
    # Paramters
    total_angle = 0
    # Default values
    projection = start_projection
    best_projection = None
    best_projected_data = None
    direction = np.zeros(X.shape[1])
    max_dip = 0
    while True:
        gradients, dip_value, projected_data = _dip_gradient(X, projection, check_duplicates)
        if dip_value > max_dip:
            max_dip = dip_value
            best_projection = projection
            best_projected_data = projected_data
        # Normally there is only one gradient. There can be multiple if gradient is ambiguous
        if len(gradients) > 1:
            tmp_max_dip = 0
            for tmp_gradient in gradients:
                tmp_direction = momentum * direction + step_size * tmp_gradient
                tmp_projection = projection + tmp_direction
                projected_data = np.matmul(X, tmp_projection)
                tmp_dip_value = dip_test(projected_data, just_dip=True, is_data_sorted=False)
                if tmp_dip_value > tmp_max_dip:
                    tmp_max_dip = tmp_dip_value
                    gradient = tmp_gradient
        else:
            gradient = gradients[0]

        direction = momentum * direction + step_size * gradient
        new_projection = projection + direction
        new_angle = _angle(projection, new_projection)
        projection = new_projection / np.linalg.norm(new_projection)
        if np.isnan(new_angle):
            print("Angle is NaN")
            new_angle = 0.1
        total_angle = total_angle + new_angle
        if (new_angle <= 0.2 and np.linalg.norm(direction) < 0.2) or total_angle > 360:
            break
    return max_dip, best_projection, best_projected_data


def _angle(v, w):
    quotient = np.linalg.norm(v, ord=2) * np.linalg.norm(w, ord=2)
    if quotient != 0:
        a = v.dot(w) / quotient
        if a > 1:
            a = 1
        if a < -1:
            a = -1
        theta = np.arccos(a)
    else:
        theta = 0
    return 180 * theta / np.pi


class DipExt():

    def __init__(self, n_components=None, do_dip_scaling=True, step_size=0.1, momentum=0.95, dip_threshold=0.5,
                 n_starting_points=None, check_duplicates=False):
        self.n_components = n_components
        self.do_dip_scaling = do_dip_scaling
        self.step_size = step_size
        self.momentum = momentum
        self.dip_threshold = dip_threshold
        self.n_starting_points = n_starting_points
        self.check_duplicates = check_duplicates

    def fit_transform(self, X):
        if self.n_starting_points is None:
            n_starting_points = int(np.log(X.shape[1])) + 1
        else:
            n_starting_points = self.n_starting_points
        subspace, dip_values = _dip_ext(X, self.n_components, self.do_dip_scaling, self.step_size, self.momentum,
                                        self.dip_threshold, n_starting_points, self.check_duplicates)
        self.n_components = len(dip_values)
        self.dip_values_ = dip_values
        return subspace

class DipInit(DipExt, BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters, n_components=None, do_dip_scaling=True, step_size=0.1, momentum=0.95, dip_threshold=0.5,
                 n_starting_points=None, check_duplicates=False):
        super().__init__(n_components, do_dip_scaling, step_size, momentum, dip_threshold, n_starting_points,
                         check_duplicates)
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        subspace = self.fit_transform(X)
        labels, centers = _dip_init(subspace, self.n_clusters)
        self.labels_ = labels
        self.cluster_centers_ = centers
        return self
