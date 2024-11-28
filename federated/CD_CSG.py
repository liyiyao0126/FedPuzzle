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

    def predict_graph(self, data, graph):
        """Predict from an undirected graph.

        Args:
            data (np.ndarray): A np.array of variables in the causal skeleton.
            graph (np.ndarray): The undirected graph of the causal skeleton.
        Returns:
            output(nx.DiGraph): The predicted causal DAG.
            adj(np.ndarray): The adjacent matrix of the causal DAG.
        """
        data = np.array(data)
        adj, csgset, adjset = self.get_causalstargraph(data, graph)
        output = nx.from_numpy_array(adj, create_using=nx.DiGraph)
        return output, adj, csgset, adjset

    def get_stargraph(self, graph, n):
        g = graph.to_undirected()
        A = np.array(nx.adjacency_matrix(g).todense())
        adj = np.array(np.where(A[n, :] == 1)).reshape(-1, )
        return adj

    def get_causalstargraph(self, data, graph):
        D = data.shape[1]
        adj = np.zeros([D, D])
        csgset = []
        adjset = []
        # 这里开始在切割星星图以及对每一个星星图做变分推理
        for i in range(D):
            adj_i = self.get_stargraph(graph, i) #本地
            # print("star graph:", adj_i)
            csg = np.c_[data[:, adj_i], data[:, i]] #本地
            # print("csg:", csg)
            causal_dir = CSG_model(csg) #本地或云端
            csgset.append(-causal_dir + 1)  # 带方向的星形图集合，i代表是第几个变量的
            adjset.append(adj_i)  # 所有邻居的集合
            # print(causal_dir)
            adj[adj_i, i] = causal_dir #云端代码要重新写，要对本地传上来的根据个数进行判断
            adj[i, adj_i] = -causal_dir + 1
            # print("adj:", adj)
        return adj, csgset, adjset

    def get_localcausalstargraph(self, data, graph):
        D = data.shape[1]
        adj = np.zeros([D, D])
        csgset = []
        adjset = []
        # 这里开始在切割星星图以及对每一个星星图做变分推理
        for i in range(D):
            adj_i = self.get_stargraph(graph, i) #本地
            # print("star graph:", adj_i)
            csg = np.c_[data[:, adj_i], data[:, i]] #本地
            # print("csg:", csg)
            causal_dir = CSG_model(csg) #本地或云端
            # print(causal_dir)
            csgset.append(-causal_dir + 1) #带方向的星形图集合，i代表是第几个变量的
            adjset.append(adj_i) #所有邻居的集合
            # adj[adj_i, i] = causal_dir #云端代码要重新写，要对本地传上来的根据个数进行判断
            # adj[i, adj_i] = -causal_dir + 1
            # print("adj:", adj)
        return csgset, adjset

    # def get_neighbors(G, x: int, y: int):
    #     return [i for i in range(len(G)) if G[x][i] == True and i != y]
    #
    # def gauss_ci_test(suff_stat, x: int, y: int, K: List[int], cut_at: float = 0.9999999):
    #     """条件独立性检验"""
    #     C = suff_stat["C"]
    #     n = suff_stat["n"]
    #
    #     # ------ 偏相关系数 ------
    #     if len(K) == 0:  # K 为空
    #         r = C[x, y]
    #
    #     elif len(K) == 1:  # K 中只有一个点，即一阶偏相关系数
    #         k = K[0]
    #         r = (C[x, y] - C[x, k] * C[y, k]) / math.sqrt((1 - C[y, k] ** 2) * (1 - C[x, k] ** 2))
    #
    #     else:  # 其实我没太明白这里是怎么求的，但 R 语言的 pcalg 包就是这样写的
    #         m = C[np.ix_([x] + [y] + K, [x] + [y] + K)]
    #         p = np.linalg.pinv(m)
    #         r = -p[0, 1] / math.sqrt(abs(p[0, 0] * p[1, 1]))
    #
    #     r = min(cut_at, max(-cut_at, r))
    #
    #     # Fisher's z-transform
    #     z = 0.5 * math.log1p((2 * r) / (1 - r))
    #     z_standard = z * math.sqrt(n - len(K) - 3)
    #
    #     # Φ^{-1}(1-α/2)
    #     p_value = 2 * (1 - norm.cdf(abs(z_standard)))
    #
    #     return p_value
    #
    # def skeleton(suff_stat, alpha: float):
    #     n_nodes = suff_stat["C"].shape[0]
    #
    #     # 分离集
    #     O = [[[] for _ in range(n_nodes)] for _ in range(n_nodes)]
    #
    #     # 完全无向图
    #     G = [[i != j for i in range(n_nodes)] for j in range(n_nodes)]
    #
    #     # 点对（不包括 i -- i）
    #     pairs = [(i, (n_nodes - j - 1)) for i in range(n_nodes) for j in range(n_nodes - i - 1)]
    #
    #     done = False
    #     l = 0  # 节点数为 l 的子集
    #
    #     while done != True and any(G):
    #         done = True
    #
    #         # 遍历每个相邻点对
    #         for x, y in pairs:
    #             if G[x][y] == True:
    #                 neighbors = suff_stat.get_neighbors(G, x, y)  # adj(C,x) \ {y}
    #
    #                 if len(neighbors) >= l:  # |adj(C, x) \ {y}| > l
    #                     done = False
    #
    #                     # |adj(C, x) \ {y}| = l
    #                     for K in set(combinations(neighbors, l)):
    #                         # 节点 x, y 是否被节点数为 l 的子集 K d-seperation
    #                         # 条件独立性检验，返回 p-value
    #                         p_value = suff_stat.gauss_ci_test(suff_stat, x, y, list(K))
    #
    #                         # 条件独立
    #                         if p_value >= alpha:
    #                             G[x][y] = G[y][x] = False  # 去掉边 x -- y
    #                             O[x][y] = O[y][x] = list(K)  # 把 K 加入分离集 O
    #                             print("k", K)
    #                             break
    #
    #             l += 1
    #     print(4)
    #     print(np.asarray(G, dtype=int), O)
    #     return np.asarray(G, dtype=int), O