import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
        Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        n = self.adj_mat.shape[0]
        # Initialize the MST as a zero matrix
        self.mst = np.zeros_like(self.adj_mat)
        
        # Track visited vertices
        visited = [False] * n
        
        # Priority queue: (weight, from_vertex, to_vertex)
        min_heap = []
        
        # Start with vertex 0
        visited[0] = True
        
        # Add all edges from vertex 0 to the heap
        for j in range(n):
            if self.adj_mat[0, j] > 0:  # Edge exists
                heapq.heappush(min_heap, (self.adj_mat[0, j], 0, j))
        
        # Process edges until all vertices are visited
        edges_added = 0
        while min_heap and edges_added < n - 1:
            weight, u, v = heapq.heappop(min_heap)
            
            # If v is already visited, skip this edge
            if visited[v]:
                continue
            
            # Add edge to MST
            visited[v] = True
            self.mst[u, v] = weight
            self.mst[v, u] = weight  # Keep symmetric for undirected graph
            edges_added += 1
            
            # Add all edges from the newly visited vertex v
            for j in range(n):
                if not visited[j] and self.adj_mat[v, j] > 0:
                    heapq.heappush(min_heap, (self.adj_mat[v, j], v, j))
