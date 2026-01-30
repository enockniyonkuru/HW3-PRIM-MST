import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error
    
    n = adj_mat.shape[0]
    
    # Check that the MST is symmetric (undirected graph)
    assert np.allclose(mst, mst.T), 'MST adjacency matrix should be symmetric'
    
    # Check that MST has exactly n-1 edges (property of spanning trees)
    edge_count = np.sum(mst > 0) // 2  # Divide by 2 because matrix is symmetric
    assert edge_count == n - 1, f'MST should have {n-1} edges but has {edge_count}'
    
    # Check that all MST edges are present in the original graph
    for i in range(n):
        for j in range(i+1, n):
            if mst[i, j] > 0:
                assert adj_mat[i, j] > 0, f'Edge ({i},{j}) in MST but not in original graph'
                assert approx_equal(mst[i, j], adj_mat[i, j]), \
                    f'MST edge weight {mst[i, j]} differs from original {adj_mat[i, j]}'
    
    # Check total weight
    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'
    
    # Check connectivity: ensure the MST forms a connected graph using DFS
    def is_connected(adj):
        visited = [False] * n
        stack = [0]
        visited[0] = True
        count = 1
        
        while stack:
            node = stack.pop()
            for neighbor in range(n):
                if adj[node, neighbor] > 0 and not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
                    count += 1
        
        return count == n
    
    assert is_connected(mst), 'MST is not connected'


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    Unit test for MST construction on a simple 3-node triangle graph.
    Tests basic MST behavior on a small complete graph with known properties.
    
    Graph structure:
        0 -- 2 -- 1
         \       /
          \     /
           \  /
             3
    Expected MST edges: (0,2) with weight 2 and (2,1) with weight 3
    Expected MST weight: 5
    """
    # Create a 3-node complete graph with known weights
    adj_mat = np.array([
        [0, 5, 2],
        [5, 0, 3],
        [2, 3, 0]
    ], dtype=float)
    
    g = Graph(adj_mat)
    g.construct_mst()
    
    # Check MST with expected weight of 5 (minimum edges: 2 and 3)
    check_mst(g.adj_mat, g.mst, 5)
