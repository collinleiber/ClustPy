from clustpy.utils import minimum_spanning_tree_prims, reachability_distances, DCTree
import numpy as np

def test_dc_tree_min():
    X = np.array([
        [0.0, 0.0],
        [-3.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [-7.0, 0.0],
        [-8.0, 0.0]
    ])
    # Min_points = 3
    dctree = DCTree(X, min_points = 3, use_less_memory = False)
    root = dctree.root
    assert root.dist == 5.0
    assert root.left.leaves == [0, 2, 3, 1, 4]
    assert root.right.leaves == [5]
    assert root.right.dist == 0.0
    assert root.right.right is None
    assert root.right.left is None
    assert root.left.dist == 4.0
    assert root.left.left.leaves == [0, 2, 3, 1]
    assert root.left.right.leaves == [4]
    assert root.left.right.dist == 0.0
    assert root.left.right.left is None
    assert root.left.right.right is None
    assert root.left.left.dist == 4.0
    assert root.left.left.left.leaves == [0, 2, 3]
    assert root.left.left.right.leaves == [1]
    assert root.left.left.left.dist == 3.0
    assert root.left.left.left.left.leaves == [0, 2]
    assert root.left.left.left.right.leaves == [3]
    assert root.left.left.left.left.dist == 3.0
    assert root.left.left.left.left.left.leaves == [0]
    assert root.left.left.left.left.right.leaves == [2]
    all_distances = dctree.dc_distances()
    expected_distances = np.array([[0., 4., 3., 3., 4., 5.],
                 [4., 0., 4., 4., 4., 5.],
                 [3., 4., 0., 3., 4., 5.],
                 [3., 4., 3., 0., 4., 5.],
                 [4., 4., 4., 4., 0., 5.],
                 [5., 5., 5., 5., 5., 0.]
    ])
    assert np.array_equal(all_distances, expected_distances)
    all_distances_reverse = dctree.dc_distances(np.arange(6), np.arange(5)[::-1], access_method="dc_dist")
    expected_distances = expected_distances[:, np.arange(5)[::-1]]
    assert np.array_equal(all_distances_reverse, expected_distances)
    labels = dctree.get_k_center(3)
    assert np.array_equal(labels, np.array([2, 2, 2, 2, 1, 0]))
    eps = dctree.get_eps_for_k(3)
    assert np.abs(eps - 4.0) < 1e-5
    # Min_points = 2
    dctree = DCTree(X, min_points = 2, use_less_memory = True)
    root = dctree.root
    assert root.dist == 4.0
    assert root.left.leaves == [0, 2, 3, 1]
    assert root.right.leaves == [4, 5]
    assert root.right.dist == 1.0
    assert root.right.left.leaves == [4]
    assert root.right.right.leaves == [5]
    assert root.right.left.dist == 0.0
    assert root.right.left.left is None
    assert root.right.left.right is None
    assert root.right.right.dist == 0.0
    assert root.right.right.left is None
    assert root.right.right.right is None
    assert root.left.dist == 3.0
    assert root.left.left.leaves == [0, 2, 3]
    assert root.left.right.leaves == [1]
    assert root.left.right.dist == 0.0
    assert root.left.right.left is None
    assert root.left.right.right is None
    assert root.left.left.dist == 2.0
    assert root.left.left.left.leaves == [0]
    assert root.left.left.right.leaves == [2, 3]
    assert root.left.left.right.dist == 1.0
    assert root.left.left.right.left.leaves == [2]
    assert root.left.left.right.right.leaves == [3]
    all_distances = dctree.dc_distances()
    expected_distances = np.array([[0., 3., 2., 2., 4., 4.],
                 [3., 0., 3., 3., 4., 4.],
                 [2., 3., 0., 1., 4., 4.],
                 [2., 3., 1., 0., 4., 4.],
                 [4., 4., 4., 4., 0., 1.],
                 [4., 4., 4., 4., 1., 0.]
    ])
    assert np.array_equal(all_distances, expected_distances)
    all_distances_reverse = dctree.dc_distances(np.arange(6), np.arange(5)[::-1], access_method="dc_dist")
    expected_distances = expected_distances[:, np.arange(5)[::-1]]
    assert np.array_equal(all_distances_reverse, expected_distances)
    labels = dctree.get_k_center(3)
    assert np.array_equal(labels, np.array([2, 1, 2, 2, 0, 0]))
    eps = dctree.get_eps_for_k(3)
    assert np.abs(eps - 3.0) < 1e-5
    # Test different accesses to DCTree
    node = dctree[4]
    assert node.id == 4
    all_nodes = dctree[np.arange(11)]
    for i, node in enumerate(all_nodes):
        assert node.id == i
    # Test repr
    assert type(dctree.__repr__) is str
    assert type(node.__repr__) is str


def test_reachability_distances():
    X = np.array([
        [0.0, 0.0],
        [-3.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [-7.0, 0.0],
        [-8.0, 0.]
    ])
    distances = reachability_distances(X, min_points = 2)
    expected_distances = [[0., 3., 2., 3., 7., 8.],
                 [3., 0., 5., 6., 4., 5.],
                 [2., 5., 0., 1., 9., 10.],
                 [3., 6., 1., 0., 10., 11. ],
                 [7., 4., 9., 10, 0., 1.],
                 [8., 5., 10., 11., 1., 0.]
    ]
    assert np.array_equal(distances, expected_distances)
    distances = reachability_distances(X, min_points = 3)
    expected_distances = [[0., 4., 3., 3., 7., 8.],
                 [4., 0., 5., 6., 4., 5.],
                 [3., 5., 0., 3., 9., 10.],
                 [3., 6., 3., 0., 10., 11.],
                 [7., 4., 9., 10., 0., 5.],
                 [8., 5., 10., 11., 5., 0.]
    ]
    assert np.array_equal(distances, expected_distances)


def test_minimum_spanning_tree_prims():
    distance_matrix = np.array([
            [0.0, 3.0, 2.0, 3.0, 7.0, 8.0],
            [3.0, 0.0, 5.0, 6.0, 4.0, 5.0],
            [2.0, 5.0, 0.0, 1.0, 9.0, 10.0],
            [3.0, 6.0, 1.0, 0.0, 10.0, 11.0],
            [7.0, 4.0, 9.0, 10.0, 0.0, 1.0],
            [8.0, 5.0, 10.0, 11.0, 1.0, 0.0]
        ])
    mst = minimum_spanning_tree_prims(distance_matrix, use_less_memory=False, min_points=None)
    expected_mst = np.array([(0, 2, 2.0), (2, 3, 1.0), (0, 1, 3.0), (1, 4, 4.0), (4, 5, 1.0)], dtype=([("i", int), ("j", int), ("dist", float)]))
    # test with use_less_memory=True
    X = np.array([
            [0.0, 0.0],
            [-3.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [-7.0, 0.0],
            [-8.0, 0.]
        ])
    mst = minimum_spanning_tree_prims(X, use_less_memory=True, min_points=None)
    assert np.array_equal(mst, expected_mst)
    # Test reachability distance
    core_distances = [3, 4, 2, 3, 7]
    mst = minimum_spanning_tree_prims(X, use_less_memory=True, min_points=3)
    expected_mst = np.array([(0, 2, 3.0), (0, 3, 3.0), (0, 1, 4.0), (1, 4, 4.0), (1, 5, 5.0)], dtype=([("i", int), ("j", int), ("dist", float)]))
    assert np.array_equal(mst, expected_mst)
