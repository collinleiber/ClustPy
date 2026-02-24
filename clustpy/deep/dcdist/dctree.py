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

import gzip
import numpy as np
import sys

from collections import deque
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from sklearn.metrics import pairwise_distances
from typing import List, Literal, Optional, Sequence, Tuple, Union


class DCTree:
    """
    The DCTree computes the dc_distances and stores them in a tree structure.

    By using the euler tour of the tree and a sparse table for range minimum queries,
    the lca-elements in the tree (i.e. the dc_distances) can be computed in O(1) time.

    The function `dc_dist(i, j)` returns the dc_distance between `points[i]` and `points[j]`
    in O(1) time.
    The function `dc_distance(X, Y=None, access_method="tree")` returns a dc_distance matrix
    with the dc_distance from each pair of `points[X]` and `points[Y]`.
    See the section Functions for more details.

    The DCTree provides `serialize` and `serialize_compressed` functions to serialize the
    DCTree. With `deserialize` or `deserialize_compressed` the DCTree can be deserialized again.

    The DCTree provides `save` and `load` functions to save / load the DCTree to / from disk.


    Parameters
    ----------
    points : np.ndarray
        points, of which the dc_distances should be computed of.

    min_points : int, optional
        min_points parameter used for the computation of the dc_distances, by default 5.

    access_method: str
        Default access_method for dc_distances, by default "tree".

    no_fastindex: bool = False
        No fast index structure is constructed if this parameter is set to `True`.
        Disables the functions `dc_dist` and `dc_distances` with `access_method = "dc_dist"`.
        Only useful if you only use the `dc_distance` function with `access_method = "tree"`,
        e.g. for calculating all pair dc_distances with `dc_dists = dc_tree.dc_distances()`.

    use_less_memory: bool = False
        Don't precalculate the reachability distance. Saves quadratic RAM usage, but also needs
        double the time for computing.


    Functions
    ---------
    DCTree[x] or DCTree[i,j] or DCTree[X,Y]:
        Returns the _DCNode of given index if `arg` is an integer or a Sequence.

        If DCTree[i,j] is used, the dc_dist of `points[i]` and `points[j]` is returned
        (i and j are integer).

        If DCTree[X,Y] is used, the dc_dist matrix between each pair of `points[X]` and
        `points[Y]` is returned (X and Y are np.ndarray or Sequences).

    dc_dist : (self, i: int, j: int) -> distance
        returns the dc_distance between points[i] and points[j] in O(1) time.

    dc_distances : (self, X = None, Y = None, access_method = `self.accesss_method`) -> np.ndarray
        Computes the dc_distance matrix between each pair of points[X] and points[Y].
        If X=None, X=range(n) is used.
        If Y=None, Y=X is used.

        `access_method` (by default `self.accesss_method`):
            - "tree":    Traverses the tree in O(n) time (n = len(points)),
                         no matter the size of X / Y.

            - "dc_dist": Uses the dc_dist function and needs O(k*l) time
                         (k = len(X), l = len(Y))).

            "dc_dist" is often faster if X and Y are smaller than ~10% of n = len(points).

        Returns dc_dists: ndarray of shape (n_samples_X, n_samples_Y)


    serialize : (dc_tree: DCTree) -> str
        Serializes the DCTree `dc_tree` to a string.

    serialize_compressed : (dc_tree: DCTree) -> bytes
        Serializes the DCTree `dc_tree` to a compressed byte array.

    save : (dc_tree: DCTree, file_path: str) -> save to disk
        Saves the DCTree `dc_tree` to disk at `file_path`.


    deserialize : (data: str, access_method = "tree", no_fastindex = False, n_jobs = None) -> DCTree
        Deserializes a string `str` to a DCTree.

    deserialize_compressed : (data: bytes, access_method = "tree", no_fastindex = False, n_jobs = None) -> DCTree
        Deserializes a compressed byte array `bytes` to a DCTree.

    load : (file_path: str, access_method = "tree", no_fastindex = False, n_jobs = None) -> DCTree
        Loads a DCTree from disk at `file_path`.


    Examples
    --------
    >>> import dcdist
    >>> points = np.array([[1,6],[2,6],[6,2],[14,17],[123,3246],[52,8323],[265,73]])
    >>> dc_tree = dcdist.DCTree(points, 5)
    >>> print(dc_tree.dc_dist(2,5))
    >>> print(dc_tree.dc_distances(range(len(points))))
    >>> print(dc_tree.dc_distances([0,1], [2,3]))

    >>> s = dcdist.serialize(dc_tree)
    >>> dc_tree_new = dcdist.deserialize(s)

    >>> b = dcdist.serialize_compressed(dc_tree)
    >>> dc_tree_new = dcdist.deserialize_compressed(b)

    >>> dcdist.save(dc_tree, "./data.dctree")
    >>> dc_tree_new = dcdist.load("./data.dctree")


    References
    ----------
    Anna Beer, Andrew Draganov, Ellen Hohma, Philipp Jahn, Christian M.M. Frey, and Ira Assent.
    "Connecting the Dots -- Density-Connectivity Distance unifies DBSCAN, k-Center and Spectral
    Clustering." KDD '23. 2023.
    """

    n: int  # len(points)
    min_points: int
    root: _DCNode

    euler: List[_DCNode]
    level: List[int]
    f_occur: List[int]
    level_table: _SparseTable

    access_method: str
    no_fastindex: bool
    n_jobs: int
    no_gil: bool

    def __init__(
        self,
        points: np.ndarray,
        min_points: int = 5,
        min_points_mr: Optional[int] = None,
        access_method: str = "tree",
        no_fastindex: bool = False,
        use_less_memory: bool = False,
        n_jobs: Optional[int] = None,
        precomputed=False,
    ):
        self.n = points.shape[0]
        self.min_points = min_points
        self.access_method = access_method
        self.no_fastindex = no_fastindex

        if min_points_mr is None:
            min_points_mr = min_points

        if n_jobs is None or n_jobs == 0:
            self.n_jobs = cpu_count()
        elif n_jobs < 0:
            self.n_jobs = max(cpu_count() + 1 + n_jobs, 1)
        else:
            self.n_jobs = n_jobs
        self.no_gil = sys.flags.nogil if hasattr(sys.flags, "nogil") else False

        if use_less_memory and not precomputed:
            mst_edges = self._get_mst_edges(points, use_less_memory=True)
        else:
            if not precomputed:
                reach_dists = calculate_reachability_distance(points, min_points_mr)
            else:
                reach_dists = points
            mst_edges = self._get_mst_edges(reach_dists)
            if not precomputed:
                del reach_dists
        mst_edges.sort(order="dist")
        self.root = self._build_tree(mst_edges)
        del mst_edges

        if not self.no_fastindex:
            self._init_fast_index()

    def _init_fast_index(self):
        n_nodes = 2 * self.n - 1
        self.euler = [self.root] * (2 * n_nodes - 1)
        self.level = [0] * (2 * n_nodes - 1)
        self.f_occur = [-1] * n_nodes
        self._euler_tour(self.root)
        self.level_table = _SparseTable(self.level)

    def __getitem__(
        self,
        arg: Union[
            Union[int, slice, Sequence[int], np.ndarray],
            Union[
                Tuple[int, int],
                Tuple[int, np.ndarray, Sequence[int]],
                Tuple[np.ndarray, Sequence[int], int],
                Tuple[np.ndarray, Sequence[int], np.ndarray, Sequence[int]],
            ],
        ],
    ) -> Union[Union[_DCNode, List[_DCNode]], Union[float, np.ndarray]]:
        """
        Returns the _DCNode of given index if `arg` is an integer or a Sequence.

        If DCTree[i,j] is used, the dc_dist of `points[i]` and `points[j]` is returned
        (i and j are integer).

        If DCTree[X,Y] is used, the dc_dist matrix between each pair of `points[X]` and
        `points[Y]` is returned (X and Y are np.ndarray or Sequences).
        """

        index_error_msg = f"`{arg}` needs to be an integer, Sequence, np.ndarray, tuple of integer, or tuple of np.ndarray / Sequence!"

        if isinstance(arg, tuple):
            if len(arg) != 2:
                raise IndexError(index_error_msg)
            (i, j) = arg
            if isinstance(i, int) and isinstance(j, int):
                return self.dc_dist(i, j)
            if isinstance(i, int) and isinstance(j, (np.ndarray, Sequence)):
                if isinstance(j, np.ndarray):
                    j = j.flatten()
                return self.dc_distances([i], j)
            if isinstance(i, (np.ndarray, Sequence)) and isinstance(j, int):
                if isinstance(i, np.ndarray):
                    i = i.flatten()
                return self.dc_distances(i, [j])
            elif isinstance(i, (np.ndarray, Sequence)) and isinstance(j, (np.ndarray, Sequence)):
                if isinstance(i, np.ndarray):
                    i = i.flatten()
                if isinstance(j, np.ndarray):
                    j = j.flatten()
                return self.dc_distances(i, j)
            else:
                raise IndexError(index_error_msg)

        elif isinstance(arg, int):
            return self.euler[self.f_occur[arg]]
        elif isinstance(arg, (Sequence, np.ndarray)):
            return [self.euler[self.f_occur[i]] for i in arg]
        elif isinstance(arg, slice):
            return [self.euler[i] for i in self.f_occur[arg]]

        else:
            raise IndexError(index_error_msg)

    def __repr__(self):
        if self.root is None:
            return ""

        # pointer_right = "└──"
        # pointer_left = "├──" if self.root.right else "└──"
        pointer_right = "   "
        pointer_left = "   " if self.root.right else "   "
        return (
            f"{self.root}"
            f"{self.__repr__help(self.root.left, pointer_left, '', self.root.right is not None)}"
            f"{self.__repr__help(self.root.right, pointer_right, '', False)}"
        )

    def __repr__help(self, node: Optional[_DCNode], pointer: str, padding: str, has_right_sibling: bool):
        if node is None:
            return ""

        # padding_for_both = padding + ("|  " if has_right_sibling else "   ")
        # pointer_right = "└──"
        # pointer_left = "├──" if node.right else "└──"
        padding_for_both = padding + ("   " if has_right_sibling else "   ")
        pointer_right = "   "
        pointer_left = "   " if node.right else "   "
        return (
            f"\n   {padding.replace('|', ' ')}// #region"
            f"\n{padding}{pointer}{node}"
            f"{self.__repr__help(node.left, pointer_left, padding_for_both, node.right is not None)}"
            f"{self.__repr__help(node.right, pointer_right, padding_for_both, False)}"
            f"\n   {padding.replace('|', ' ')}// #endregion"
        )

    def dc_dist(self, i: int, j: int) -> float:
        """Returns the dc_distance from points[i] to points[j] in O(1) time."""
        if i == j:
            return 0
        return self._lca(i, j).dist

    def _lca(self, i: int, j: int) -> _DCNode:
        first_i = self.f_occur[i]
        first_j = self.f_occur[j]
        if first_i > first_j:
            first_i, first_j = first_j, first_i
        return self.euler[self.level_table.query(first_i, first_j)]

    def dc_distances(
        self,
        X: Union[Sequence[int], np.ndarray, None] = None,
        Y: Union[Sequence[int], np.ndarray, None] = None,
        access_method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Computes the dc_distance matrix between each pair of points[X] and points[Y].
        If X=None, X=range(n) is used.
        If Y=None, Y=X is used.

        `access_method` (by default `self.access_method`):
            - "tree":    traverses the tree in O(n) time (n = len(points)),
                         no matter the size of X / Y.
            - "dc_dist": uses the dc_dist function and needs O(k*l) time
                         (k = len(X), l = len(Y))).

            "dc_dist" is often faster if X and Y are smaller than ~10% of n = len(points).

        Returns dc_dists: ndarray of shape (n_samples_X, n_samples_Y)
        """

        if X is None:
            X = range(self.n)

        if Y is None:
            Y = X

        if access_method is None:
            access_method = self.access_method

        if access_method == "dc_dist" and self.no_fastindex:
            print("No fastindex computed. Fallback to access_method: `tree`.")
            access_method = "tree"

        if access_method == "dc_dist" and not self.no_fastindex:
            dc_dists = np.zeros((len(X), len(Y)))

            for i in range(len(X)):
                for j in range(i + 1 if X is Y else 0, len(Y)):
                    if X[i] != Y[j]:
                        dc_dists[i, j] = self.dc_dist(X[i], Y[j])
            if X is Y:
                dc_dists = dc_dists + dc_dists.T
            return dc_dists

        elif access_method == "tree":
            dc_dists = np.zeros((len(X), len(Y)))

            idx_rev_X = np.full(self.n, -1)
            idx_rev_X[X] = range(len(X))

            if X is Y:
                idx_rev_Y = idx_rev_X
            else:
                idx_rev_Y = np.full(self.n, -1)
                idx_rev_Y[Y] = range(len(Y))

            inodes = self.get_internal_nodes()
            inodes_start = 0
            inodes_end = len(inodes)

            n_jobs = self.n_jobs if self.no_gil else 4
            pool = ThreadPool(n_jobs)
            n_inodes = self.n - 1
            chunk_size = int(np.ceil(n_inodes / n_jobs))
            start_points = range(inodes_start, inodes_end, chunk_size)

            def func(start):
                for i in range(start, min(start + chunk_size, inodes_end)):
                    inode = inodes[i]
                    # inode always has left and right node
                    i_leaves = inode.left.leaves
                    j_leaves = inode.right.leaves

                    (i_, j_) = (idx_rev_X[i_leaves], idx_rev_Y[j_leaves])
                    (i_, j_) = (i_[i_ != -1], j_[j_ != -1])
                    dc_dists[(i_[:, np.newaxis], j_[np.newaxis, :])] = inode.dist

                    if X is Y:
                        dc_dists[(j_[:, np.newaxis], i_[np.newaxis, :])] = inode.dist
                    else:
                        (i_, j_) = (idx_rev_Y[i_leaves], idx_rev_X[j_leaves])
                        (i_, j_) = (i_[i_ != -1], j_[j_ != -1])
                        dc_dists[(j_[:, np.newaxis], i_[np.newaxis, :])] = inode.dist

            pool.map(func, start_points)
            return dc_dists

        raise ValueError(f"'{access_method}' is no valid `access_method`")

    def _get_mst_edges(self, dist_matrix: np.ndarray, use_less_memory: bool = False) -> np.ndarray:
        """Prim's algorithm to build up the minimum spanning tree in O(n^2) time."""
        # dist_matrix are the points, if use_less_memory=True
        n = self.n
        nodes_min_dist = np.full(n, np.inf)
        parent = np.full(n, 0)
        not_in_mst = np.full(n, True)
        u = 0  # Start node
        nodes_min_dist[u] = 0
        not_in_mst[u] = False
        mst_edges = np.empty((n - 1), dtype=([("i", int), ("j", int), ("dist", float)]))

        if use_less_memory:
            reach_dists = np.empty((dist_matrix.shape[0]))
            for i in range(dist_matrix.shape[0]):
                eucl_dists = np.linalg.norm(dist_matrix - dist_matrix[i], axis=1)
                reach_dists[i] = np.max(np.partition(eucl_dists, self.min_points)[: self.min_points])

        for i in range(n - 1):
            if use_less_memory:
                dist_u = np.linalg.norm(dist_matrix - dist_matrix[u], axis=1)  # Euclidean distances
                dist_u = np.maximum(dist_u, reach_dists[u])  # reachability distance of i
                dist_u = np.maximum(dist_u, reach_dists)  # reachability distance of all points

                v = np.where(not_in_mst & (dist_u < nodes_min_dist))[0]
                nodes_min_dist[v] = dist_u[v]
            else:
                v = np.where(not_in_mst & (dist_matrix[u] < nodes_min_dist))[0]
                nodes_min_dist[v] = dist_matrix[u, v]
            parent[v] = u

            arg = np.argmin(nodes_min_dist[not_in_mst])
            u = np.arange(n)[not_in_mst][arg]
            mst_edges[i] = (parent[u], u, nodes_min_dist[u])
            not_in_mst[u] = False

        return mst_edges

    def _build_tree(self, mst_edges: np.ndarray) -> _DCNode:
        """Kruskal's algorithm to build up the DCTree with the precomputed
        and sorted mst_edges in O(n) time."""
        union_find = _UnionFind(self.n)
        node = _DCNode(id=0, dist=0.0, leaves=[0])
        idx = self.n
        k = len(mst_edges)
        for i, j, dist in mst_edges:
            i_root = union_find.find(i)
            j_root = union_find.find(j)
            node = _DCNode(
                id=idx,
                dist=dist,
                k=-1 if (len(i_root.leaves) + len(j_root.leaves)) < self.min_points else k,
                left=i_root,
                right=j_root,
                leaves=i_root.leaves + j_root.leaves,
            )
            i_root.parent = node
            j_root.parent = node
            union_find.union(i, j, node)
            idx += 1
            if not node.k == -1:
                k -= 1
        return node

    def _euler_tour(self, root: _DCNode):
        """Euler tour to get the euler, level, and f_occur lists in O(n) time."""
        DOWN, UP = 0, 1
        stack: deque[Tuple[_DCNode, int, Literal[0, 1]]] = deque([(root, 0, DOWN)])  # (node, level, DOWN / UP)

        pos = 0
        while len(stack):
            (node, level, status) = stack.pop()

            if status == DOWN:
                if self.f_occur[node.id] == -1:
                    self.f_occur[node.id] = pos

                self.euler[pos] = node
                self.level[pos] = level
                pos += 1

                if node.right:
                    stack.append((node, level, UP))
                    stack.append((node.right, level + 1, DOWN))
                if node.left:
                    stack.append((node, level, UP))
                    stack.append((node.left, level + 1, DOWN))

            elif status == UP:
                self.euler[pos] = node
                self.level[pos] = level
                pos += 1

    def get_internal_nodes(self):
        nodes = []
        stack: deque[_DCNode] = deque([self.root])

        while len(stack):
            node = stack.pop()

            if node and node.left and node.right:
                nodes.append(node)

            if node.left:
                stack.append(node.left)

            if node.right:
                stack.append(node.right)
        return nodes

    def traverse_until_k(self, k):
        from queue import PriorityQueue

        result_nodes = set([self.root])
        if k == 1:
            return result_nodes

        stack = PriorityQueue()
        stack.put((-self.root.dist, self.root, self.root))
        while not stack.empty():
            _, node, parent_node = stack.get()

            if node is None:
                return

            if node.k >= 0:
                if node.left.k != -1:
                    if node.left.k != -1 and node.right.k != -1:
                        result_nodes.discard(parent_node)
                        result_nodes.discard(node)
                        result_nodes.add(node.left)
                    if node.parent.left.k == -1 or node.parent.right.k == -1:
                        stack.put((-node.left.dist, node.left, parent_node))
                    else:
                        stack.put((-node.left.dist, node.left, node))
                if node.right.k != -1:
                    if node.left.k != -1 and node.right.k != -1:
                        result_nodes.discard(parent_node)
                        result_nodes.discard(node)
                        result_nodes.add(node.right)
                    if node.parent.left.k == -1 or node.parent.right.k == -1:
                        stack.put((-node.right.dist, node.right, parent_node))
                    else:
                        stack.put((-node.right.dist, node.right, node))

            if len(result_nodes) >= k:
                break

        return result_nodes

    def get_k_center(self, k):
        nodes = self.traverse_until_k(k)
        labels = np.full(self.n, -1)
        for i, node in enumerate(nodes):
            labels[np.array(node.leaves)] = i
        return labels

    def get_eps_for_k(self, k, eps=-3e-12):
        nodes = self.traverse_until_k(k)
        min_eps = np.inf
        for node in nodes:
            min_eps = min(min_eps, node.parent.dist)
        return min_eps + eps


def _serialize(root: _DCNode):
    tree_list = []
    stack: deque[Optional[_DCNode]] = deque([root])
    while len(stack):
        node = stack.pop()
        if node is None:
            tree_list.append("/")
        elif node.left is None and node.right is None:
            tree_list.append(f"{node.id}|{node.dist}")
        else:
            tree_list.append(f"'{node.id}|{node.dist}")
            stack.append(node.right)
            stack.append(node.left)
    return ",".join(tree_list)


def serialize(dc_tree: DCTree) -> str:
    """Serializes the DCTree `dc_tree` to a string."""

    sep = "<\1dc_tree>"
    # Tree structure, min_points, n_jobs, no_gil
    return _serialize(dc_tree.root) + sep + str(dc_tree.min_points)


def serialize_compressed(dc_tree: DCTree) -> bytes:
    """Serializes the DCTree `dc_tree` to a gzip-compressed byte array."""
    data = serialize(dc_tree)
    byte_data = bytes(data, "utf-8")
    return gzip.compress(byte_data)


def save(dc_tree: DCTree, file_path: str) -> None:
    """Saves the DCTree `dc_tree` to disk at `file_path`."""
    byte_data = serialize_compressed(dc_tree)
    with open(file_path, "wb") as file:
        file.write(byte_data)


def _deserialize(data: List[str]) -> _DCNode:
    id, dist = data[0].split("|")
    root = _DCNode(id=int(id[1:]), dist=float(dist), leaves=[])

    DOWN, UP = 0, 1
    stack: deque[Literal[0, 1]] = deque([UP, DOWN, DOWN])
    nodes: deque[_DCNode] = deque([root])
    res: deque[Optional[_DCNode]] = deque([])

    pos = 0
    while len(stack):
        status = stack.pop()

        if status == DOWN:
            pos += 1
            if data[pos] == "/":
                res.append(None)
                continue
            id, dist = data[pos].split("|")
            if id[0] != "'":
                res.append(_DCNode(id=int(id), dist=0, leaves=[int(id)]))
                continue
            inode = _DCNode(id=int(id[1:]), dist=float(dist), leaves=[])
            nodes.append(inode)
            stack.extend([UP, DOWN, DOWN])

        elif status == UP:
            node = nodes.pop()
            node.right = res.pop()
            node.left = res.pop()
            res.append(node)
            node.leaves = node.left.leaves + node.right.leaves
    return root


def deserialize(
    data: str,
    access_method: Optional[str] = None,
    no_fastindex: bool = False,
    n_jobs: Optional[int] = None,
) -> DCTree:
    """Deserializes a string `str` to a DCTree."""
    dc_tree = object.__new__(DCTree)

    sep = "<\1dc_tree>"
    # Tree structure, min_points, n_jobs, no_gil
    (tree_data, min_points) = data.split(sep)
    dc_tree.min_points = int(min_points)
    dc_tree.access_method = access_method if access_method is not None else "tree"
    dc_tree.n_jobs = n_jobs if n_jobs else cpu_count()
    dc_tree.no_gil = sys.flags.nogil if hasattr(sys.flags, "nogil") else False

    tree_data_list = tree_data.split(",")
    dc_tree.root = _deserialize(tree_data_list)
    dc_tree.n = (len(tree_data_list) + 1) // 2

    dc_tree.no_fastindex = no_fastindex
    if not dc_tree.no_fastindex:
        dc_tree._init_fast_index()
    return dc_tree


def deserialize_compressed(
    compressed_data: bytes,
    access_method: Optional[str] = None,
    no_fastindex: bool = False,
    n_jobs: Optional[int] = None,
) -> DCTree:
    """Deserializes a compressed byte array `bytes` to a DCTree."""
    byte_data = gzip.decompress(compressed_data)
    data = str(byte_data, "utf-8")
    return deserialize(data, access_method, no_fastindex, n_jobs=n_jobs)


def load(
    file_path: str,
    access_method: Optional[str] = None,
    no_fastindex: bool = False,
    n_jobs: Optional[int] = None,
) -> DCTree:
    """Loads a DCTree from disk at `file_path`."""
    file = open(file_path, "rb")
    byte_data = file.read()
    file.close()
    return deserialize_compressed(byte_data, access_method, no_fastindex, n_jobs=n_jobs)


def calculate_reachability_distance(
    points: np.ndarray, min_points: int = 5, n_jobs: Optional[int] = None
) -> np.ndarray:
    """
    Calculates the reachability distance of points using the min_points threshold in O(n^2) time.

    Raises a ValueError if min_points is larger than the number of points.
    """

    if min_points > points.shape[0]:
        raise ValueError(f"Min points ({min_points}) can't exceed the size of the dataset ({points.shape[0]})")

    eucl_dists = pairwise_distances(points, metric="euclidean", n_jobs=n_jobs)

    if min_points > 1:
        # Get reachability for each point with respect to min_points parameter
        reach_dists = np.empty((points.shape[0]))
        for i in range(points.shape[0]):
            reach_dists[i] = np.max(np.partition(eucl_dists[i], min_points)[:min_points])
        np.maximum(eucl_dists, reach_dists[np.newaxis, :], eucl_dists)
        np.maximum(eucl_dists, reach_dists[:, np.newaxis], eucl_dists)
        np.fill_diagonal(eucl_dists, 0)
    return eucl_dists


class _DCNode:
    id: int
    dist: float
    leaves: List[int]
    left: Optional[_DCNode]
    right: Optional[_DCNode]
    parent: _DCNode
    k: int

    def __init__(
        self,
        id: int,
        dist: float,
        leaves: List[int],
        left: Optional[_DCNode] = None,
        right: Optional[_DCNode] = None,
        parent=None,
        k=-1,
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
        self.k = k

    def __repr__(self):
        return f"DCNode #{self.id} ({self.dist}) - {self.k}"

    def __lt__(self, other):
        return self.dist < other.dist


class _UnionFind:
    """
    UnionFind structure which provides the functions `find` and `union` with amortized
    time complexity of O(α(n)), where α(n) is the inverse Ackermann function.
    """

    root: List[_DCNode]
    parent: List[int]
    rank: List[int]

    def __init__(self, n: int):
        self.root = [_DCNode(id=i, dist=0, leaves=[i]) for i in range(n)]
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, x: int) -> _DCNode:
        """Finds the representative node of x."""
        return self.root[self._find(x)]

    def _find(self, x: int) -> int:
        """Finds the representative of x."""
        parent = self.parent[x]
        if parent != x:
            parent = self._find(parent)
            self.parent[x] = parent
        return parent

    def union(self, x: int, y: int, node: _DCNode):
        """Union the set which contains x with the set which contains y."""
        xset = self._find(x)
        yset = self._find(y)
        if xset == yset:
            return

        # Put smaller ranked item under bigger ranked item if ranks are different
        if self.rank[xset] < self.rank[yset]:
            self.parent[xset] = yset
            self.root[yset] = node
        elif self.rank[xset] > self.rank[yset]:
            self.parent[yset] = xset
            self.root[xset] = node

        # If ranks are same, then move y under x (doesn't matter which one goes where)
        # and increment rank of x's tree
        else:
            self.parent[yset] = xset
            self.root[xset] = node
            self.rank[xset] = self.rank[xset] + 1


class _SparseTable:
    """
    SparseTable structure which provides the function `query` which finds the index of the
    minimum value within the range [l,r] in the array `arr`.
    Needs O(n * logn) time and storage to precompute and O(1) for the query.
    """

    # arr: List[int]
    arr: np.ndarray
    # sparse_table: List[List[int]]
    sparse_table: np.ndarray
    pow_2: List[int]
    log_2: List[int]

    def __init__(self, arr: List[int]):
        self.arr = np.asarray(arr, dtype=np.int64)
        n = len(arr)
        log_n = n.bit_length() - 1
        # self.sparse_table = [[-1] * log_n for _ in range(n - 1)]
        self.sparse_table = np.full((n, log_n), -1, dtype=np.int64)

        self.pow_2 = [1 << i for i in range(log_n + 1)]
        self.log_2 = [i.bit_length() - 1 for i in range(n)]

        # for i in range(0, n - 1):
        #     self.sparse_table[i][0] = i if self.arr[i] < self.arr[i + 1] else i + 1
        idx = np.arange(n - 1, dtype=np.int64)
        self.sparse_table[: n - 1, 0] = np.where(self.arr[idx] < self.arr[idx + 1], idx, idx + 1)

        for j in range(1, log_n):
            step = self.pow_2[j]  # 2**j
            limit = n - self.pow_2[j + 1] + 1  # number of start positions
            left_idx = self.sparse_table[:limit, j - 1]  # L for i = 0..limit‑1
            right_idx = self.sparse_table[step : step + limit, j - 1]  # R for i = 0..limit‑1
            choose_left = self.arr[left_idx] <= self.arr[right_idx]

            self.sparse_table[:limit, j] = np.where(choose_left, left_idx, right_idx)

            # for i in range(n - self.pow_2[j + 1] + 1):
            #     L = self.sparse_table[i][j - 1]
            #     R = self.sparse_table[i + step][j - 1]
            #     self.sparse_table[i][j] = L if arr[L] <= arr[R] else R

    def query(self, l: int, r: int) -> int:
        # assert L > R, f"L value ({L}) needs to be larger than R value ({R})"
        if l == r:
            return l
        k = self.log_2[r - l + 1]
        l_k = self.sparse_table[l][k - 1]
        r_k = self.sparse_table[r - self.pow_2[k] + 1][k - 1]
        return l_k if self.arr[l_k] <= self.arr[r_k] else r_k
