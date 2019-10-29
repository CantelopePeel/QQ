from qiskit.transpiler import CouplingMap

coupling_edges1 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]

coupling_edges2 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
                   [1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]]

coupling_map1 = CouplingMap(coupling_edges1)
print(coupling_map1.shortest_undirected_path())