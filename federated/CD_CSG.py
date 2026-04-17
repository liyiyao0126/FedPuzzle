from cdt.data import AcyclicGraphGenerator
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
from GenLearning import CSG_model
# from itertools import combinations
# from scipy.stats import norm
# import math
# from typing import List

class CD_CSG():
    """Causal discovery via causal star graphs.

    **Description**: Causal discovery via causal star graphs （CD-CSG） is causal discovery
    framework to learn causal directed acyclic graphs (DAGs).
    It bases on the generalized learning and identify the causal directions through
    finding the asymmetry in the forward and backward model of CD-CSG.

    **Data Type**: Continuous

    Example:
        >>> import networkx as nx
        >>> import matplotlib.pyplot as plt
        >>> from cdt.data import AcyclicGraphGenerator
        >>> from GenLearning import CSG_model
        >>> generator = AcyclicGraphGenerator('polynomial', noise='gaussian', noise_coeff=0.4, npoints=500, nodes=8)
        >>> data, graph1 = generator.generate()
        >>> graph = graph1.to_undirected() # get the causal skeleton of graph1

        >>> obj = CD_CSG()
        >>> # This example uses the predict_graph() method
        >>> DAG, adj = obj.predict_graph(data, graph)

        >>> # This example uses the predict_adj() method
        >>> A = np.array(nx.adj_matrix(graph).todense()) # get the adjacent matrix of the graph
        >>> DAG, adj = obj.predict_adj(data, A)

        >>> # To view the result
        >>> plt.figure(figsize=(8, 4))
        >>> plt.subplot(121)
        >>> plt.title('ground truth')
        >>> nx.draw(graph1, pos=nx.circular_layout(graph1), node_color='g', edge_color='r', with_labels=True, font_size=18, width=2, node_size=1000)
        >>> plt.subplot(122)
        >>> plt.title('CSG result')
        >>> nx.draw(DAG, pos=nx.circular_layout(DAG), with_labels=True, font_size=18, width=2, node_size=1000)
        >>> plt.show()
    """

    def predict_adj(self, data, ADJ):
        """Predict from an adjacent matrix.

        Args:
            data (np.ndarray): A np.array of variables in the causal skeleton.
            ADJ (np.ndarray): The adjacent matrix of the causal skeleton.
        Returns:
            output(nx.DiGraph): The predicted causal DAG.
            adj(np.ndarray): The adjacent matrix of the causal DAG.
        """
        graph = nx.from_numpy_array(ADJ)
        data = np.array(data)
        adj= self.get_causalstargraph(data, graph)
        output = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        return output, adj

    def predict_graph(self, data, graph, delta_threshold, true_adj_matrix=None,result_filename= None):
        """Predict from an undirected graph.

        Args:
            data (np.ndarray): A np.array of variables in the causal skeleton.
            graph (np.ndarray): The undirected graph of the causal skeleton.
        Returns:
            output(nx.DiGraph): The predicted causal DAG.
            adj(np.ndarray): The adjacent matrix of the causal DAG.
        """
        data = np.array(data)
        adj, csgset, adjset = self.get_causalstargraph(data, graph,delta_threshold, true_adj_matrix,result_filename)
        output = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        return output, adj, csgset, adjset

    def get_stargraph(self, graph, n):
        g = graph.to_undirected()
        A = np.array(nx.adjacency_matrix(g).todense())
        adj = np.array(np.where(A[n, :] == 1)).reshape(-1, )
        return adj

    def get_causalstargraph(self, data, graph,delta_threshold, true_adj_matrix=None,result_filename= None):
        D = data.shape[1]
        adj = np.zeros([D, D])
        csgset = []
        adjset = []
        star_graph_metrics = []
        for i in range(D):
            adj_i = self.get_stargraph(graph, i)
            # print("star graph:", adj_i)
            csg = np.c_[data[:, adj_i], data[:, i]]
            # print("csg:", csg)
            causal_dir = CSG_model(csg,delta_threshold)
            csgset.append(-causal_dir + 1)
            adjset.append(adj_i)
            # print(causal_dir)
            adj[adj_i, i] = causal_dir
            adj[i, adj_i] = -causal_dir + 1
            # print("adj:", adj)
            if true_adj_matrix is not None:
                direction_errors = 0
                total_edges = len(adj_i)

                for idx, neighbor in enumerate(adj_i):
                    if true_adj_matrix[neighbor, i] == 1:
                        if causal_dir[idx] < 0.5:
                            direction_errors += 1
                    elif true_adj_matrix[i, neighbor] == 1:
                        if causal_dir[idx] > 0.5:
                            direction_errors += 1

                accuracy = (total_edges - direction_errors) / total_edges if total_edges > 0 else 0
                star_graph_metrics.append({
                    'node': i,
                    'direction_errors': direction_errors,
                    'total_edges': total_edges,
                    'accuracy': accuracy
                })

        if star_graph_metrics:
            total_direction_errors = sum([m['direction_errors'] for m in star_graph_metrics])
            total_edges = sum([m['total_edges'] for m in star_graph_metrics])
            avg_accuracy = np.mean([m['accuracy'] for m in star_graph_metrics]) if star_graph_metrics else 0

            with open(result_filename, 'a', encoding='utf-8') as f:
                f.write(f"\nStar Graph Metrics:\n")
                f.write(f"  Total Direction Errors (SHD): {total_direction_errors}\n")
                f.write(f"  Total Edges: {total_edges}\n")
                f.write(f"  Average Accuracy (F1): {avg_accuracy:.4f}\n")
                f.write("  Per-node Details:\n")
                for metric in star_graph_metrics:
                    f.write(f"    Node {metric['node']}: Errors={metric['direction_errors']}, "
                            f"Total={metric['total_edges']}, Accuracy={metric['accuracy']:.4f}\n")
                f.write("====================\n")
                f.write(f"csgset:\n {csgset}")
                f.write(f"adjset:\n {adjset}")


            pass
        return adj, csgset, adjset

    def get_localcausalstargraph(self, data, graph):
        D = data.shape[1]
        adj = np.zeros([D, D])
        csgset = []
        adjset = []
        for i in range(D):
            adj_i = self.get_stargraph(graph, i)
            # print("star graph:", adj_i)
            csg = np.c_[data[:, adj_i], data[:, i]]
            # print("csg:", csg)
            causal_dir = CSG_model(csg)
            # print(causal_dir)
            csgset.append(-causal_dir + 1)
            adjset.append(adj_i)
        return csgset, adjset