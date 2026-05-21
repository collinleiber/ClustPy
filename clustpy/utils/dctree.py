"""
@authors:
Pascal Weber
"""

# Implementation of the dc-distance with a DCTree by
# - Author: Pascal Weber
# - Source: https://github.com/pasiweber/SHADE

# Paper: Connecting the Dots -- Density-Connectivity Distance unifies DBSCAN, k-Center and Spectral Clustering
# Authors: Anna Beer, Andrew Draganov, Ellen Hohma, Philipp Jahn, Christian M.M. Frey, and Ira Assent
# Link: https://doi.org/10.1145/3580305.3599283


from __future__ import annotations
import numpy as np
from typing import List, Optional, Sequence, Tuple, Union
from scipy.spatial.distance import pdist, cdist, squareform


class DCTree:
    """
    The DCTree computes the dc_distances and stores them in a tree structure.
    By using the euler tour of the tree and a sparse table for range minimum queries,
    the lca-elements in the tree (i.e. the dc_distances) can be computed in O(1) time.
    The function `dc_dist(i, j)` returns the dc_distance between `points[i]` and `points[j]`
    in O(1) time.
    The function `dc_distance(idx_X, idx_Y=None, access_method="tree")` returns a dc_distance matrix
    with the dc_distance from each pair of `points[idx_X]` and `points[idx_Y]`.
    See the section Functions for more details.

    Parameters
    ----------
    X : np.ndarray
        points, of which the dc_distances should be computed of.
    min_points : int, optional
        min_points parameter used for the computation of the dc_distances (default: 5).
    use_less_memory: bool
      Use less memory when constructing the DCTree.
      This will, however, increase the runtime (default: False).

    Functions
    ---------
    DCTree[x]:
        Returns the _DCNode of given index.
    dc_dist : (i, j) -> distance
        returns the dc_distance between points[i] and points[j] in O(1) time.
    dc_distances : (idx_X, idx_Y) -> np.ndarray
        Computes the dc_distance matrix between each pair of points[idx_X] and points[idx_Y].
        Returns dc_dists as ndarray of shape (n_samples_X, n_samples_Y)

    Examples
    --------
    >>> points = np.array([[1,6],[2,6],[6,2],[14,17],[123,3246],[52,8323],[265,73]])
    >>> dc_tree = dcdist.DCTree(points, 5)
    >>> print(dc_tree.dc_dist(2,5))
    >>> print(dc_tree.dc_distances(range(len(points))))
    >>> print(dc_tree.dc_distances([0,1], [2,3]))

    References
    ----------
    Anna Beer, Andrew Draganov, Ellen Hohma, Philipp Jahn, Christian M.M. Frey, and Ira Assent.
    "Connecting the Dots -- Density-Connectivity Distance unifies DBSCAN, k-Center and Spectral
    Clustering." KDD '23. 2023.
    """

    def __init__(
        self,
        X: np.ndarray,
        min_points: int = 5,
        use_less_memory: bool = False
    ):
        self.n = X.shape[0]
        self.min_points = min_points
        if not use_less_memory:
            # Calculate pair-wise reachability distance
            X = reachability_distances(X, min_points)
        mst_edges = minimum_spanning_tree_prims(X, use_less_memory=use_less_memory, min_points=min_points)
        self.root = self._build_tree(mst_edges)
        self._init_fast_index()

    def _build_tree(self, mst_edges: np.ndarray) -> _DCNode:
        """
        Build the final DCTree using the MST edges.

        Parameters
        ----------
        mst_edges : np.ndarray
            The edges of the MST, including the dc-distances

        Returns
        -------
        new_root_node : _DCNode
            The root node of the tree
        """
        mst_edges.sort(order="dist")
        # Create leaf nodes
        leaf_nodes = [_DCNode(idx, 0.0, [idx]) for idx in range(self.n)]
        idx = self.n
        for i, j, dist in mst_edges:
            root_i = self._get_root(leaf_nodes[i])
            root_j = self._get_root(leaf_nodes[j])
            new_leaves = root_i.leaves + root_j.leaves
            new_root_node = _DCNode(
                id=idx,
                dist=dist,
                left=root_i,
                right=root_j,
                leaves=new_leaves,
            )
            root_i.parent = new_root_node
            root_i.root = new_root_node
            root_j.parent = new_root_node
            root_j.root = new_root_node
            idx += 1
        return new_root_node

    def _get_root(self, node: _DCNode) -> '_DCNode':
        """
        Get the root of a node and update all root_ entries that have been visited.

        Parameters
        ----------
        node : _DCNode
            The input node

        Returns
        -------
        root_node : _DCNode
            The root node
        """
        root_node = node
        while root_node.root is not None:
            root_node = root_node.root
        # Override old roots
        if root_node is not node:
            current_node = node
            while current_node.root != root_node:
                next_node = current_node.root
                current_node.root = root_node
                current_node = next_node
        return root_node

    def _init_fast_index(self) -> None:
        """
        Build a fast index structure for the DCTree.
        """
        n_nodes = 2 * self.n - 1
        self.euler = []
        self.level = []
        self.f_occur = [None] * n_nodes
        # Euler tour to get the euler, level, and f_occur lists in O(n) time.
        DOWN, UP = 0, 1
        stack = [(self.root, 0, DOWN)]  # (node, level, DOWN / UP)
        while len(stack) > 0:
            (node, level, status) = stack.pop()
            if status == DOWN:
                if self.f_occur[node.id] is None:
                    self.f_occur[node.id] = len(self.euler)
                self.euler.append(node)
                self.level.append(level) 
                if node.right is not None:
                    stack.append((node, level, UP))
                    stack.append((node.right, level + 1, DOWN))
                if node.left is not None:
                    stack.append((node, level, UP))
                    stack.append((node.left, level + 1, DOWN))
            elif status == UP:
                self.euler.append(node)
                self.level.append(level)
        # SparseTable structure which finds the index of the minimum value within the range [l,r] in self.level
        levels = np.array(self.level, dtype=int)
        n_elements = len(levels)
        log_n = n_elements.bit_length()
        self.sparse_table = np.zeros((log_n, n_elements), dtype=int)
        self.sparse_table[0] = np.arange(n_elements)
        for j in range(1, log_n):
            stride = 1 << (j - 1)
            limit = n_elements - (1 << j) + 1
            if limit <= 0:
                break
            left_indices = self.sparse_table[j - 1, :limit]
            right_indices = self.sparse_table[j - 1, stride : stride + limit]
            choose_left = levels[left_indices] <= levels[right_indices]
            self.sparse_table[j, :limit] = np.where(choose_left, left_indices, right_indices)

    def __getitem__(
        self,
        point_idx: Union[int, Sequence[int], np.ndarray]
    ) -> Union[_DCNode, List[_DCNode]]:
        """
        Returns the _DCNode of given index if `point_idx` is an integer or a list of _DCNodes if `point_idx' is a Sequence.

        Parameters
        ----------
        point_idx : Union[int, Sequence[int], np.ndarray]
            The input parameter as described above

        Returns
        -------
        result : Union[_DCNode, List[_DCNode]]
            The querried nodes in the tree
        """
        if isinstance(point_idx, int):
            return self.euler[self.f_occur[point_idx]]
        elif isinstance(point_idx, (Sequence, np.ndarray)):
            return [self.euler[self.f_occur[i]] for i in point_idx]
        else:
            raise IndexError(f"`{point_idx}` needs to be an integer, Sequence or np.ndarray!")

    def dc_dist(self, i: int, j: int) -> float:
        """
        Returns the dc_distance between points[i] and points[j] in O(1) time.
        
        Parameters
        ----------
        i : int
            index of the first point
        j : int
            index of the second point
        
        Returns
        -------
        dc_distance : float
            The dc_distance of the two points
        """
        if i == j:
            return 0.0
        l = self.f_occur[i]
        r = self.f_occur[j]
        if l > r:
            l, r = r, l
        k = (r - l + 1).bit_length() - 1
        idx_l = self.sparse_table[k, l]
        idx_r = self.sparse_table[k, r - (1 << k) + 1]
        ancestor = self.euler[idx_l if self.level[idx_l] <= self.level[idx_r] else idx_r]
        dc_distance = ancestor.dist
        return dc_distance

    def dc_distances(
        self,
        idx_X: Union[Sequence[int], np.ndarray, None] = None,
        idx_Y: Union[Sequence[int], np.ndarray, None] = None,
        access_method: str = "tree",
    ) -> np.ndarray:
        """
        Computes the dc_distance matrix between each pair of points[idx_X] and points[idx_Y].
        If idx_X=None, idx_X=range(n) is used.
        If idx_Y=None, idx_Y=idx_X is used.

        Parameters
        ----------
        idx_X : Union[Sequence[int], np.ndarray, None]
            the first set of indices
        idx_Y : Union[Sequence[int], np.ndarray, None]
            the second set of indices
        access_method : str
            "tree":     traverses the tree in O(n) time (n = len(points)), no matter the size of X / Y.
            "dc_dist":  uses the dc_dist function and needs O(k*l) time (k = len(X), l = len(Y))).
            (default: "tree")
        
        Returns
        -------
        dc_dists : np.array
            ndarray of shape (n_samples_X, n_samples_Y) containing the distances
        """
        if idx_X is None:
            idx_X = range(self.n)
        if idx_Y is None:
            idx_Y = idx_X
        dc_dists = np.zeros((len(idx_X), len(idx_Y)))
        if access_method == "dc_dist":
            # Get distances from the index structure
            for i in range(len(idx_X)):
                start_index = i + 1 if idx_X is idx_Y else 0
                for j in range(start_index, len(idx_Y)):
                    dc_dists[i, j] = self.dc_dist(idx_X[i], idx_Y[j])
        elif access_method == "tree":
            # Get distances from the tree
            idx_rev_X = np.full(self.n, -1, dtype=int)
            idx_rev_X[idx_X] = range(len(idx_X))
            if idx_X is idx_Y:
                idx_rev_Y = idx_rev_X
            else:
                idx_rev_Y = np.full(self.n, -1, dtype=int)
                idx_rev_Y[idx_Y] = range(len(idx_Y))
            node_list = [self.root]
            while len(node_list) > 0:
                node = node_list.pop()
                if node.left is not None:
                    node_list.append(node.left)
                if node.right is not None:
                    node_list.append(node.right)
                if node.left is not None and node.right is not None:
                    i_leaves = node.left.leaves
                    j_leaves = node.right.leaves
                    (i_, j_) = (idx_rev_X[i_leaves], idx_rev_Y[j_leaves])
                    (i_, j_) = (i_[i_ != -1], j_[j_ != -1])
                    if len(i_) > 0 and len(j_) > 0:
                        dc_dists[(i_[:, np.newaxis], j_[np.newaxis, :])] = node.dist
                    # In case of asymmetric call
                    if idx_X is not idx_Y:
                        (i_, j_) = (idx_rev_Y[i_leaves], idx_rev_X[j_leaves])
                        (i_, j_) = (i_[i_ != -1], j_[j_ != -1])
                        if len(i_) > 0 and len(j_) > 0:
                            dc_dists[(j_[:, np.newaxis], i_[np.newaxis, :])] = node.dist
        else:
            raise ValueError(f"'{access_method}' is no valid `access_method`")
        if idx_X is idx_Y:
            # Mirror values
            dc_dists = dc_dists + dc_dists.T
        return dc_dists


    def _traverse_until_k_clusters(self, n_clusters: int) -> List[_DCNode]:
        """
        Traverse the tree to identify n_clusters nodes that minimize the maximum within-cluster distance.

        Parameters
        ----------
        n_clusters : int
            the number of desired clusters

        Returns
        -------
        result_nodes : List[_DCNode]
            List of nodes, where each node represents the root of a cluster
        """
        assert n_clusters <= self.n, "n_clusters can not be larger than the number of input points"
        node_list = [self.root]
        result_nodes = []
        id_threshold = 2 * self.n - n_clusters - 1
        while len(node_list) > 0:
            node = node_list.pop()
            if node.id <= id_threshold:
                result_nodes.append(node)
            else:
                if node.left is not None:
                    node_list.append(node.left)
                if node.right is not None:
                    node_list.append(node.right)
        assert len(result_nodes) == n_clusters, "Length of the result_nodes list is not equal to k"
        return result_nodes

    def get_k_center(self, n_clusters: int) -> np.ndarray:
        """
        Extract the k center approach.
        It identifies clusters such that the maximum distance within a cluster is minimized.

        Parameters
        ----------
        n_clusters : int
            the number of desired clusters

        Returns
        -------
        labels : np.ndarray
            The cluster labels
        """
        nodes = self._traverse_until_k_clusters(n_clusters)
        labels = np.zeros(self.n, dtype=np.int32)
        for i, node in enumerate(nodes):
            labels[np.array(node.leaves)] = i
        return labels

    def get_eps_for_k(self, n_clusters: int, eps: float = 1e-10):
        """
        Receive the largest possible eps value such that one receives exectly n_clusters.

        Parameters
        ----------
        n_clusters : int
            the number of desired clusters
        eps : float
            a small constaint used to differentiate from the next potential cluster (default: 1e-10)
            
        Returns
        -------
        min_eps : float
            The resulting epsilon value
        """
        assert eps > 0, "Eps has to be a positive number"
        nodes = self._traverse_until_k_clusters(n_clusters)
        min_eps = np.inf
        for node in nodes:
            if node.parent is not None:
                min_eps = min(min_eps, node.parent.dist)
        return min_eps - eps

    def __repr__(self) -> str:
        """
        Return a string describing the DCTree.

        Returns
        -------
        repr_string : str
            The string representation of the tree
        """
        if self.root is None:
            return ""
        pointer_right = "   "
        pointer_left = "   " if self.root.right else "   "
        repr_string = (
            f"{self.root}"
            f"{self.__repr__help(self.root.left, pointer_left, '', self.root.right is not None)}"
            f"{self.__repr__help(self.root.right, pointer_right, '', False)}"
        )
        return repr_string

    def __repr__help(self, node: Optional[_DCNode], pointer: str, padding: str, has_right_sibling: bool) -> str:
        """
        Helper for the __repr__ function.

        Parameters
        ----------
        node : Optional[_DCNode]
            The current node
        pointer : str
            The pointer string
        padding : str
            The padding string
        has_right_sibling : bool
            True if node has a right sibbling

        Returns
        -------
        repr_string : str
            The string for this node
        """
        if node is None:
            return ""
        padding_for_both = padding + ("   " if has_right_sibling else "   ")
        pointer_right = "   "
        pointer_left = "   " if node.right else "   "
        repr_string = (
            f"\n   {padding.replace('|', ' ')}// #region"
            f"\n{padding}{pointer}{node}"
            f"{self.__repr__help(node.left, pointer_left, padding_for_both, node.right is not None)}"
            f"{self.__repr__help(node.right, pointer_right, padding_for_both, False)}"
            f"\n   {padding.replace('|', ' ')}// #endregion"
        )
        return repr_string


class _DCNode:
    """
    A node in a DCTree.
    Each node contains a set of leaves and can have a left and a right child node.

    Parameters
    ----------
    id : int
        the id of the node
    dist : float
        the dc distance
    leaves : List[int]
        list of the ids of its child nodes
    left : Optional[_DCNode]
        the left child node (default: None)
    right : Optional[_DCNode]
        the right child node (default: None)
    parent : Optional[_DCNode]
        the parent node. If None, it will be set to itself (default: None)
    root : Optional[_DCNode]
        the root node (default: None)
    """

    def __init__(
        self,
        id: int,
        dist: float,
        leaves: List[int],
        left: Optional[_DCNode] = None,
        right: Optional[_DCNode] = None,
        parent: Optional[_DCNode] = None,
        root: Optional[_DCNode] = None
    ):
        self.id = id
        self.dist = dist
        self.leaves = leaves
        self.left = left
        self.right = right
        if not parent:
            self.parent = self
        else:
            self.parent = parent
        self.root = root

    def __repr__(self) -> str:
        """
        Return a string containing the id and dist of this node.

        Returns
        -------
        to_str : str
            The string
        """
        to_str = f"DCNode #{self.id} ({self.dist})"
        return to_str

    def __lt__(self, other_node: _DCNode) -> bool:
        """
        Less than method of the node. Compares the dist with respect to the other input node.

        Parameters
        ----------
        other_node : _DCNode
            the other node

        Returns
        -------
        is_less : bool
            True if dist is smaller than dist of the other node
        """
        less_than = self.dist < other_node.dist
        return less_than


def reachability_distances(X: np.ndarray, min_points: int = 5) -> np.ndarray:
    """
    Calculates the reachability distances between points using the min_points threshold in O(n^2) time.

    Parameters
    ----------
    X : np.ndarray
        the points
    min_points : int, optional
        min_points parameter (default: 5)

    Returns
    -------
    reach_distances : np.ndarray
        Array containing the pair-wise reachability distances

    Raises
    ------
    Raises a ValueError if min_points is larger than the number of points.
    """
    if min_points > X.shape[0]:
        raise ValueError(f"Min points ({min_points}) can't exceed the size of the dataset ({X.shape[0]})")
    eucl_distances = squareform(pdist(X, metric="euclidean"))
    if min_points > 1:
        core_distances = np.partition(eucl_distances, min_points - 1, axis=1)[:, min_points - 1]
        reach_distances = np.maximum(eucl_distances, np.maximum.outer(core_distances, core_distances))
        np.fill_diagonal(reach_distances, 0)
    else:
        reach_distances = eucl_distances
    return reach_distances


def minimum_spanning_tree_prims(matrix: np.ndarray, use_less_memory: bool = False, min_points: int = None) -> np.ndarray:
    """
    Create a Minimum-spanning-tree of a given matrix using Prim's algorithm.
    The tree will be build in O(n^2) time.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix
    use_less_memory : bool
        If true, the MST will not directly be build for the input matrix but the matrix will be used to construct a distance matrix first.
        Saves quadratic RAM usage, but also needs double the time for computing (default: False)
    min_points : int
        Min_points for calculating the reachability distance. Only relevant if use_less_memory is True.
        If min_points is None, the euclidean distance will be used (default: None)

    Returns
    -------
    mst_edges : np.ndarray
        The edges of the Minimum-spanning-tree, represented as a (n-1, 3) matrix with entries corresponding to (node_i, node_j, dist_ij)
    """
    assert (matrix.shape[0] == matrix.shape[1]) or use_less_memory, "Input matrix must be quadratic or use_less_memory must be True."
    n = matrix.shape[0]
    nodes_min_dist = np.full(n, np.inf)
    parent = np.zeros(n, dtype=int)
    not_in_mst = np.ones(n, dtype=bool)
    mst_edges = np.empty((n - 1), dtype=([("i", int), ("j", int), ("dist", float)]))
    # If min_points is not None, use reachability distance => calculate core distances of all points
    if use_less_memory and min_points is not None:
        core_distances = np.zeros(n)
        for i in range(n):
            eucl_distances = cdist([matrix[i]], matrix, metric="euclidean").ravel()
            core_distances[i] = np.partition(eucl_distances, min_points - 1)[min_points - 1]
    # Start building the MST
    u = 0
    nodes_min_dist[u] = 0
    not_in_mst[u] = False
    for i in range(n - 1):
        if use_less_memory:
            eucl_distances = cdist([matrix[u]], matrix, metric="euclidean").ravel()
            if min_points is None:
                dist_u = eucl_distances
            else:
                dist_u = np.maximum(eucl_distances, np.maximum(core_distances[u], core_distances))
        else:
            # If use_less_memory=False, 'matrix' is expected to be the precomputed distance matrix
            dist_u = matrix[u]
        update_mask = not_in_mst & (dist_u < nodes_min_dist)
        # Update distances and parents
        nodes_min_dist[update_mask] = dist_u[update_mask]
        parent[update_mask] = u
        # Select next closest unvisited node
        masked_dists = np.where(not_in_mst, nodes_min_dist, np.inf)
        u = np.argmin(masked_dists)
        mst_edges[i] = (parent[u], u, nodes_min_dist[u])
        not_in_mst[u] = False
    return mst_edges
