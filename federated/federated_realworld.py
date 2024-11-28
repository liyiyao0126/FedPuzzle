
import logging
import os, sys
import pickle

import networkx as nx
import torch
import numpy as np
import shutil

from typing import Dict, List

from matplotlib import pyplot as plt

sys.path.append("../")
from federated.utils import calculate_metrics, find_shortest_distance_dict
from federated.logging_settings import logger
from federated.causal_learning import ENCOAlg
from client.causal_graphs.graph_definition import CausalDAGDataset
from client.causal_discovery.utils import find_best_acyclic_graph

from cdt.metrics import SHD
from glob import glob
from client.causal_graphs.graph_real_world import load_graph_file


class FederatedSimulator:
    """
    Design a simulation for learning causal graphs in federated setup.

    A simple experimental run could be as follows.

        $ interventions_dict = {0: [c for c in range(15)], 1: [c for c in range(15, 30)]}
        $ federated_model = FederatedSimulator(interventions_dict)
        $ federated_model.initialize_clients()
        $ federated_model.execute_simulation()

    """

    def __init__(self, accessible_interventions: Dict[int, List[int]],
                 num_rounds: int = 5, num_clients: int = 2, experiment_id: int = 0,
                 repeat_id: int = 0, output_dir: str = 'default_federated_experiment',
                 client_parallelism: bool = False, verbose: bool = False):
        """ Initialize a federated setup for simulation.

        Args:
            accessible_interventions(Dict[int, List[int]]): A dictionary containing the number of
                intervened variables for each client. The key is client id.
            num_rounds (int, optional): Total number of federated rounds. Defaults to 5.
            num_clients (int, optional): Number of clients collaborating in the simulation. Defaults to 2.
            experiment_id (int, optional): Unique id of this experiment. Defaults to 0.
            repeat_id (int, optional): Number of random seeds for each simulation. Defaults to 0.
            output_dir (str, optional): Directory for saving the results. Defaults to
                'default_federated_experiment'.

            client_parallelism (bool, optional): Set True if you have enough GPU to give each client one.
                Defaults to False.
            verbose (bool, optional): Set True to see more detailed output. Defaults to False.
        """

        self.prior_theta = None
        self.prior_gamma = None

        self.global_dataset_dag = None
        self.__num_rounds = num_rounds
        assert num_rounds > 0, "Number of rounds should be at least 1."

        self.__num_clients = num_clients
        assert num_clients > 0, "Number of clients cannot be lower than 1."

        self.__experiment_id = experiment_id
        self.__repeat_id = repeat_id
        self.__output_dir = output_dir
        os.makedirs(self.__output_dir, exist_ok=True)

        self.__num_vars = 0
        self.__clients : List[ENCOAlg] = list()
        self.__client_parallelism = client_parallelism
        self.__interventions_dict = accessible_interventions
        assert len(self.__interventions_dict.keys()) == self.__num_clients, \
            "Insufficient accessible interventions info."

        if verbose: logger.setLevel(logging.DEBUG)

        self.results: Dict[str, List] = dict()
        self.initialize_results_dict()

    def initialize_clients_data(self, graph_type: str = "chain", num_vars = 30,
                                accessible_data_percentage: int = 100,
                                feature_missing_dict: Dict[int, List[int]] = None,
                                num_vars_dict: Dict[int, int] = None,
                                obs_data_size: int = 20000, int_data_size: int = 2000,
                                edge_prob: float or None = None, seed: int = 0,
                                external_global_dataset: CausalDAGDataset = None):
        """ Initialize client and clients' data for the number of clients in the federated setup.

        Args:
            graph_type (str, optional): Type of the graph. Defaults to "chain".
            num_vars (int, optional): Size of the graph. Defaults to 30.
            accessible_data_percentage (int, optional): The amount of local dataset that clients can see.
                Defaults to 100.

            obs_data_size (int, optional): Global observational dataset size. Defaults to 100000.
            int_data_size (int, optional): Global interventional dataset size. Defaults to 20000.
            edge_prob (floatorNone, optional): Edge existence probability only for random graphs.
                Defaults to None.
            seed (int, optional): Define a random seed for the dataset and graph generation.
                Defaults to 0.
        """
        self.feature_missing_dict = feature_missing_dict
        # handle real-world data as an externally initialized dataset
        if external_global_dataset is not None:
            global_dataset_dag = external_global_dataset
            self.__num_vars = external_global_dataset.adj_matrix.shape[0]
            self.prior_theta: np.ndarray = np.zeros(shape=(self.__num_vars, self.__num_vars))
            self.prior_gamma: np.ndarray = np.zeros(shape=(self.__num_vars, self.__num_vars))
            logger.info(f'External dataset parsed.')

        else:
            self.__num_vars = num_vars
            self.prior_theta: np.ndarray = np.zeros(shape=(self.__num_vars, self.__num_vars))
            self.prior_gamma: np.ndarray = np.zeros(shape=(self.__num_vars, self.__num_vars))
            global_dataset_dag = ENCOAlg.build_global_dataset(obs_data_size, int_data_size,
                                                            num_vars, graph_type, edge_prob=edge_prob,
                                                            seed=seed)
        self.global_dataset_dag = global_dataset_dag
        # print("global_dataset_dag.adj_matrix",global_dataset_dag.adj_matrix)
        # num_vars_dict = num_vars_dict if num_vars_dict is not None else {i: num_vars for i in range(self.__num_clients)}
        for client_id in range(self.__num_clients):
            try:
                enco_module = ENCOAlg(client_id, global_dataset_dag, accessible_data_percentage,
                                      self.__num_clients, self.__interventions_dict[client_id],feature_missing_dict = feature_missing_dict, num_vars_dict=num_vars_dict)
            except ValueError:
                logger.error(f'Global dataset missing for client {client_id}!')
                return

            self.__clients.append(enco_module)

    def initialize_results_dict(self):
        """Initializes the dictionary containing final results and per-round results.
        """

        self.results['round_adjs'] = list()
        self.results['round_gammas'] = list()
        self.results['round_thetas'] = list()
        self.results['round_metrics'] = list()
        self.results['round_acycle_adjs'] = list()
        self.results['round_acycle_metrics'] = list()

        self.results.update({f'client_{client_id}_metrics_acycle': list() for client_id in range(self.__num_clients)})
        self.results.update({f'client_{client_id}_metrics': list() for client_id in range(self.__num_clients)})
        self.results.update({f'client_{client_id}_adjs': list() for client_id in range(self.__num_clients)})

    def execute_simulation(self, aggregation_method: str = "naive", num_epochs: int = 2,
                           **kwargs):
        """ Execute the simulation based on the pre-defined federated setup.

        Args:
            aggregation_method (str, optional): Type of aggregation. Right now "locality", "naive"
                are implemented. Defaults to "naive".
            num_epochs (int, optional): Number of epochs for the local learning method.
                Defaults to 2.
            kwargs (dict, optinal):
                Any other argument that should be passed to the aggregation function.
        """


        prior_gamma: np.ndarray = None
        prior_theta: np.ndarray = None

        """ Federated loop """
        for round_id in range(self.__num_rounds):
            logger.info(f'Initiating round {round_id} of federated setup')

            """ Inference stage"""
            self.infer_local_models(prior_gamma, prior_theta, num_epochs, round_id)

            """ Aggregation stage """
            agg_gamma, agg_theta = self.aggregate_clients_updates(aggregation_method, round_id, **kwargs)

            """ Store round results """
            self.update_results(agg_gamma, agg_theta)

            """ Incorporate beliefs"""

            prior_gamma, prior_theta = agg_gamma, agg_theta

        """ Save the final results """
        self.save_results()

        logger.info(f'Finishing experiment {self.__experiment_id}\n')

    def infer_local_models(self, prior_gamma: np.ndarray, prior_theta: np.ndarray, num_epochs, round_id):
        """Execute the local learning methods for all clients.

        Note: Higher levels of parallelism are possible by defining client_parallelism in the instantiation step.

        Args:
            prior_gamma (np.ndarray): Prior for edge existence matrix.
            prior_theta (np.ndarray): Prior for edge orientation matrix.
            num_epochs (int): Number of epochs for ENCO.
        """

        if self.__client_parallelism:
            setup_cache_path = os.path.join(self.__output_dir, '.mpcache', f'res-{self.__experiment_id}')
            os.makedirs(setup_cache_path, exist_ok=True)

            clients_processes = list()
            for client in self.__clients:
                prior_gamma1 = prior_gamma
                prior_theta1 = prior_theta
                if round_id != 0:
                    print("prior_gamma_shape", prior_gamma.shape)
                    prior_gamma1 = np.delete(prior_gamma1, self.feature_missing_dict[client.get_client_id()], axis=0)
                    prior_gamma1 = np.delete(prior_gamma1, self.feature_missing_dict[client.get_client_id()], axis=1)
                    prior_theta1 = np.delete(prior_theta1, self.feature_missing_dict[client.get_client_id()], axis=0)
                    prior_theta1 = np.delete(prior_theta1, self.feature_missing_dict[client.get_client_id()], axis=1)
                gpu_name = f'cuda:{client.get_client_id()}'
                setup_cache_file = os.path.join(setup_cache_path, f'{id(client)}.pickle')
                client_process = torch.multiprocessing.Process(target=client.infer_causal_structure,
                                                               args=(round_id, prior_gamma1, prior_theta1, num_epochs, gpu_name,
                                                                     setup_cache_file,))
                clients_processes.append(client_process)

            for client_p in clients_processes: client_p.start()
            for client_p in clients_processes: client_p.join()

            for client in self.__clients:
                setup_cache_file = os.path.join(setup_cache_path, f'{id(client)}.pickle')
                client.retrieve_results(setup_cache_file)
        else:
            for client in self.__clients:
                feature_missing = self.feature_missing_dict[client.get_client_id()]
                prior_gamma_client = prior_gamma
                prior_theta_client = prior_theta
                if len(feature_missing) != 0 and isinstance(prior_theta, np.ndarray) and isinstance(prior_gamma,
                                                                                                    np.ndarray):
                    # for feature_missing_var in feature_missing:
                    prior_gamma_client = np.delete(prior_gamma_client, feature_missing, axis=0)
                    prior_gamma_client = np.delete(prior_gamma_client, feature_missing, axis=1)
                    prior_theta_client = np.delete(prior_theta_client, feature_missing, axis=0)
                    prior_theta_client = np.delete(prior_theta_client, feature_missing, axis=1)
                client.infer_causal_structure(round_id,prior_gamma_client, prior_theta_client, num_epochs)
    def aggregate_clients_updates(self, aggregation_method, round_id, **kwargs):
        """Perform aggregation step for all clients.

        Args:
            aggregation_method (str): Can be naive or locality aggregation so far.
            round_id (int): Current roung id.

        Returns:
            np.ndarray, np.ndarray: Aggregated gamma and theta matrices.
        """

        if aggregation_method == "naive":
            agg_gamma, agg_theta = self.naive_aggregation(round_id=round_id)
        if aggregation_method == "naive_num":
            agg_gamma, agg_theta = self.naive_aggregation_num(round_id=round_id)

        return agg_gamma, agg_theta

    def insert_column(self, A, j, new_column):
        """在矩阵A的第j列插入新列new_column"""
        left = A[:, :j]  # 取第j列之前的列
        right = A[:, j:]  # 取第j列之后的列
        shape = (A.shape[0], A.shape[1] + 1)  # 新矩阵的形状
        result = np.zeros(shape, dtype=A.dtype)  # 初始化新矩阵
        result[:, :j] = left
        result[:, j] = new_column.flatten()  # 确保新列为一维数组
        result[:, j + 1:] = right
        return result

    def insert_row(self, A, i, new_row):
        """在矩阵A的第i行插入新行new_row"""
        top = A[:i, :]  # 取第i行之前的行
        bottom = A[i:, :]  # 取第i行之后的行
        return np.vstack((top, new_row, bottom))

    def get_row_minus_column(self, matrix, i, j):
        """
        获取矩阵matrix的第i行，并从中移除第j列的元素。
        """
        # 获取第i行
        row = matrix[i, :]
        # 从该行中移除第j列的元素
        # 注意：NumPy中列索引是从0开始的
        row_without_jth_element = np.delete(row, j)
        return row_without_jth_element

    def naive_aggregation(self, round_id):
        """ Naive aggregation based on simple averaging and size of local dataset.

        Returns:
            numpy.ndarray: Prior for edge existence probabilities.
            numpy.ndarray: Prior for edge orientation probabilites.
        """

        weights: np.ndarray = np.zeros(shape=(self.__num_vars, 1))
        accumulated_gamma_mat: np.ndarray = np.zeros(shape=(self.__num_vars, self.__num_vars))
        accumulated_theta_mat: np.ndarray = np.zeros(shape=(self.__num_vars, self.__num_vars))

        csgset = []
        adjset = []

        for client in self.__clients:
            inferred_existence_mat = client.inferred_existence_mat
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_existence_mat = self.insert_row(inferred_existence_mat, self.feature_missing_dict[client.get_client_id()][i], self.get_row_minus_column(self.prior_gamma,self.feature_missing_dict[client.get_client_id()][i],self.feature_missing_dict[client.get_client_id()]))
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_existence_mat = self.insert_column(inferred_existence_mat, self.feature_missing_dict[client.get_client_id()][i], self.prior_gamma[:,self.feature_missing_dict[client.get_client_id()][i]])
            weighted_gamma_mat = inferred_existence_mat * client.get_accessible_percentage()
            accumulated_gamma_mat += weighted_gamma_mat
        #
            inferred_orientation_mat = client.inferred_orientation_mat
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_orientation_mat = self.insert_row(inferred_orientation_mat, self.feature_missing_dict[client.get_client_id()][i], self.get_row_minus_column(self.prior_theta,self.feature_missing_dict[client.get_client_id()][i],self.feature_missing_dict[client.get_client_id()]))
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_orientation_mat = self.insert_column(inferred_orientation_mat, self.feature_missing_dict[client.get_client_id()][i], self.prior_theta[:,self.feature_missing_dict[client.get_client_id()][i]])
            weighted_theta_mat = inferred_orientation_mat * client.get_accessible_percentage()
            accumulated_theta_mat += weighted_theta_mat
            weights += client.get_accessible_percentage()
            if len(self.feature_missing_dict[client.get_client_id()]) != 0:
                for feature_missing_var in self.feature_missing_dict[client.get_client_id()]:
                    weights[feature_missing_var]-=100
            csgset.append(client.csgset)
            adjset.append(client.adjset)
        accumulated_mat = np.zeros_like(accumulated_theta_mat)
        print(accumulated_gamma_mat)
        print(accumulated_theta_mat)
        print(csgset)
        print(adjset)
        prior_gamma: np.ndarray = accumulated_gamma_mat / weights
        prior_theta: np.ndarray = accumulated_theta_mat / weights
        binary_matrix_gamma = (prior_gamma > 0).astype(int)
        binary_matrix_theta = (prior_theta > 0).astype(int)
        binary_matrix = binary_matrix_gamma * binary_matrix_theta
        undirect = np.maximum(binary_matrix, binary_matrix.T)
        for client in self.__clients:
            for i in range(len(adjset[client.get_client_id()])):
                for j in range(len(adjset[client.get_client_id()][i])):
                    # print(adjset[client.get_client_id()][i][j])
                    # print(self.feature_missing_dict[client.get_client_id()][0])
                    for k in range(len(self.feature_missing_dict[client.get_client_id()])):

                        if adjset[client.get_client_id()][i][j] > self.feature_missing_dict[client.get_client_id()][k]:
                            adjset[client.get_client_id()][i][j] += 1

        for client in self.__clients:
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                csgset[client.get_client_id()].insert(self.feature_missing_dict[client.get_client_id()][i], np.array([]))
                adjset[client.get_client_id()].insert(self.feature_missing_dict[client.get_client_id()][i], np.array([]))
        # print("accumulated_mat", accumulated_mat)
        for i in range(len(undirect)):
            for j in range(len(undirect[i])):
                if undirect[i][j]>0:
                    k = 0
                    count = 0
                    length = 0
                    for client in self.__clients:

                        if j in client.adjset[i]:
                            count = count + 1
                            print("j在i的星形图中的第index个: ", np.where(client.adjset[i]==j))
                            # print(np.where(client.adjset[i]==j)[0][0])
                            k = k + client.csgset[i][np.where(client.adjset[i]==j)[0][0]] * len(client.adjset[i])
                            length = length + len(client.adjset[i])
                        if i in client.adjset[j]:
                            count = count + 1
                            print("i在j的星形图中的第index个: ", np.where(client.adjset[j] == i))
                            # print(np.where(client.adjset[i]==j)[0][0])
                            k = k - (client.csgset[j][np.where(client.adjset[j] == i)[0][0]] + 1) * len(client.adjset[j])
                            length = length - len(client.adjset[j])

                    print("第"+str(i)+"个节点到第"+str(j)+"个节点的边：" + str(k))
                    if k>0:
                        # print("self.__num_clients/2="+str(self.__num_clients/2))
                        accumulated_mat[i][j] = 1
                        accumulated_mat[j][i] = 0
                    else:
                        accumulated_mat[i][j] = 0
                        accumulated_mat[j][i] = 1
        #DAG = nx.DiGraph()
        print("accumulated_mat", accumulated_mat)

        G = nx.DiGraph()
        G.add_nodes_from([v.name for v in self.global_dataset_dag.variables])
        edges = [[self.global_dataset_dag.variables[v_idx].name for v_idx in e] for e in
                 self.global_dataset_dag.edges.tolist()]
        G.add_edges_from(edges)
        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.title('ground truth')
        nx.draw(G, pos=nx.circular_layout(G), with_labels=True)
        adj_matrix = nx.adjacency_matrix(G)
        print("原始图")
        print(G)
        print(adj_matrix)
        # print("SHD" + str(round_id) + str(SHD(accumulated_mat, self.__clients[0]._local_dag_dataset.adj_matrix)))
        # print("SHD" + str(round_id) + str(SHD(accumulated_mat, adj_matrix)))



        DAG = nx.from_numpy_array(accumulated_mat, create_using=nx.DiGraph())
        print("DAG", DAG)
        # plt.figure(figsize=(8, 4))
        plt.subplot(122)
        plt.title('result' + str(round_id))
        nx.draw(DAG, pos=nx.circular_layout(DAG), with_labels=True)
        plt.savefig("result\\result" + str(round_id) + ".png")

        f = open('result\\result.txt', 'a', encoding='utf-8')  # a 追加模式
        f.write("\n\nSHD" + str(round_id) +": "+ str(SHD(accumulated_mat, adj_matrix.toarray())))
        # f.write("\nSHD" + str(round_id) + str(SHD(accumulated_mat, self.__clients[0]._local_dag_dataset.adj_matrix)))
        f.close()


        print("accumulated_mat: ", accumulated_mat)
        print("torch.from_numpy(accumulated_mat: ",torch.from_numpy(accumulated_mat))
        print("self.global_dataset_dag.adj_matrix: ", self.global_dataset_dag.adj_matrix)
        print("torch.from_numpy(self.global_dataset_dag.adj_matrix): ",torch.from_numpy(self.global_dataset_dag.adj_matrix))
        accumulated_mat_tensor = torch.from_numpy(accumulated_mat)
        adj_matrix_tensor = torch.from_numpy(self.global_dataset_dag.adj_matrix)
        false_positives = torch.logical_and(torch.from_numpy(accumulated_mat).bool(), ~torch.from_numpy(self.global_dataset_dag.adj_matrix).bool())
        false_negatives = torch.logical_and(~torch.from_numpy(accumulated_mat).bool(), torch.from_numpy(self.global_dataset_dag.adj_matrix).bool())
        TP = torch.logical_and(torch.from_numpy(accumulated_mat).bool(), torch.from_numpy(self.global_dataset_dag.adj_matrix).bool()).float().sum().item()
        # TN = torch.logical_and(~accumulated_mat, ~self._local_dag_dataset.adj_matrix).float().sum().item()
        FP = false_positives.float().sum().item()
        FN = false_negatives.float().sum().item()
        # TN = TN - self.gamma.shape[-1]  # Remove diagonal as those are not being predicted
        recall = TP / max(TP + FN, 1e-5)
        precision = TP / max(TP + FP, 1e-5)
        f1_score = 2 * precision * recall / (precision + recall)  # F1
        orient_TP = torch.logical_and(adj_matrix_tensor == 1, accumulated_mat_tensor == 1).float().sum().item()
        orient_FN = torch.logical_and(adj_matrix_tensor == 1, accumulated_mat_tensor == 0).float().sum().item()
        orient_acc = orient_TP / max(1e-5, orient_TP + orient_FN)
        f = open('result\\result.txt', 'a', encoding='utf-8')  # a 追加模式
        f.write("\nTP" + str(round_id) +": "+ str(TP))
        f.write("\nFN" + str(round_id) +": "+ str(FN))
        f.write("\nFP" + str(round_id) +": "+ str(FP))
        f.write("\nf1_score" + str(round_id) +": "+ str(f1_score))
        f.write("\nrecall" + str(round_id) +": "+ str(recall))
        f.write("\nprecision" + str(round_id) +": " + str(precision))
        f.write("\norient_acc" + str(round_id) +": " + str(orient_acc))
        f.write("\norient_TP" + str(round_id) + ":" + str(orient_TP))
        f.write("\norient_FN" + str(round_id) + ":" + str(orient_FN))
        f.close()





        self.prior_gamma = prior_gamma
        self.prior_theta = prior_theta
        return prior_gamma, prior_theta


    def naive_aggregation_num(self, round_id):
        """ Naive aggregation based on simple averaging and size of local dataset.

        Returns:
            numpy.ndarray: Prior for edge existence probabilities.
            numpy.ndarray: Prior for edge orientation probabilites.
        """

        weights: np.ndarray = np.zeros(shape=(self.__num_vars, 1))
        accumulated_gamma_mat: np.ndarray = np.zeros(shape=(self.__num_vars, self.__num_vars))
        accumulated_theta_mat: np.ndarray = np.zeros(shape=(self.__num_vars, self.__num_vars))

        csgset = []
        adjset = []
        for client in self.__clients:
            inferred_existence_mat = client.inferred_existence_mat
            print("prior_gamma:\n", self.prior_gamma)
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_existence_mat = self.insert_row(inferred_existence_mat, self.feature_missing_dict[client.get_client_id()][i], self.get_row_minus_column(self.prior_gamma,self.feature_missing_dict[client.get_client_id()][i],self.feature_missing_dict[client.get_client_id()]))
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_existence_mat = self.insert_column(inferred_existence_mat, self.feature_missing_dict[client.get_client_id()][i], self.prior_gamma[:,self.feature_missing_dict[client.get_client_id()][i]])
            weighted_gamma_mat = inferred_existence_mat * client.get_accessible_percentage()
            accumulated_gamma_mat += weighted_gamma_mat

            inferred_orientation_mat = client.inferred_orientation_mat
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_orientation_mat = self.insert_row(inferred_orientation_mat, self.feature_missing_dict[client.get_client_id()][i], self.get_row_minus_column(self.prior_theta,self.feature_missing_dict[client.get_client_id()][i],self.feature_missing_dict[client.get_client_id()]))
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_orientation_mat = self.insert_column(inferred_orientation_mat, self.feature_missing_dict[client.get_client_id()][i], self.prior_theta[:,self.feature_missing_dict[client.get_client_id()][i]])
            weighted_theta_mat = inferred_orientation_mat * client.get_accessible_percentage()
            accumulated_theta_mat += weighted_theta_mat

            weights += client.get_accessible_percentage()

            csgset.append(client.csgset)
            adjset.append(client.adjset)
        accumulated_mat = np.zeros_like(accumulated_theta_mat)
        print(accumulated_gamma_mat)
        print(csgset)
        print(adjset)
        prior_gamma: np.ndarray = accumulated_gamma_mat / weights
        prior_theta: np.ndarray = accumulated_theta_mat / weights
        binary_matrix_gamma = (prior_gamma > 0).astype(int)
        binary_matrix_theta = (prior_theta > 0).astype(int)
        binary_matrix = binary_matrix_gamma * binary_matrix_theta
        undirect = np.maximum(binary_matrix, binary_matrix.T)
        for client in self.__clients:
            for i in range(len(adjset[client.get_client_id()])):
                for j in range(len(adjset[client.get_client_id()][i])):
                    # print(adjset[client.get_client_id()][i][j])
                    # print(self.feature_missing_dict[client.get_client_id()][0])
                    for k in range(len(self.feature_missing_dict[client.get_client_id()])):

                        if adjset[client.get_client_id()][i][j] > self.feature_missing_dict[client.get_client_id()][k]:
                            adjset[client.get_client_id()][i][j] += 1

        for client in self.__clients:
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                csgset[client.get_client_id()].insert(self.feature_missing_dict[client.get_client_id()][i], np.array([]))
                adjset[client.get_client_id()].insert(self.feature_missing_dict[client.get_client_id()][i], np.array([]))
        # print("accumulated_mat", accumulated_mat)
        for i in range(len(undirect)):
            for j in range(len(undirect[i])):
                if accumulated_gamma_mat[i][j] > 0:
                    k = 0
                    count = 0
                    for client in self.__clients:

                        if j in client.adjset[i]:
                            count = count + 1
                            print("j在i的星形图中的第index个: ", np.where(client.adjset[i] == j))
                            # print(np.where(client.adjset[i]==j)[0][0])
                            k = k + client.csgset[i][np.where(client.adjset[i] == j)[0][0]]
                        if i in client.adjset[j]:
                            count = count + 1
                            print("i在j的星形图中的第index个: ", np.where(client.adjset[j] == i))
                            # print(np.where(client.adjset[i]==j)[0][0])
                            k = k - client.csgset[j][np.where(client.adjset[j] == i)[0][0]] + 1

                    print("第" + str(i) + "个节点到第" + str(j) + "个节点的边：" + str(k))
                    if k > count / 2:
                        # print("self.__num_clients/2="+str(self.__num_clients/2))
                        accumulated_mat[i][j] = 1
                        accumulated_mat[j][i] = 0
                    else:
                        accumulated_mat[i][j] = 0
                        accumulated_mat[j][i] = 1
        #DAG = nx.DiGraph()
        print("accumulated_mat", accumulated_mat)

        G = nx.DiGraph()
        G.add_nodes_from([v.name for v in self.global_dataset_dag.variables])
        edges = [[self.global_dataset_dag.variables[v_idx].name for v_idx in e] for e in
                 self.global_dataset_dag.edges.tolist()]
        G.add_edges_from(edges)
        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.title('ground truth')
        nx.draw(G, pos=nx.circular_layout(G), with_labels=True)
        adj_matrix = nx.adjacency_matrix(G)
        print("原始图")
        print(G)
        print(adj_matrix)
        # print("SHD" + str(round_id) + str(SHD(accumulated_mat, self.__clients[0]._local_dag_dataset.adj_matrix)))
        # print("SHD" + str(round_id) + str(SHD(accumulated_mat, adj_matrix)))



        DAG = nx.from_numpy_array(accumulated_mat, create_using=nx.DiGraph())
        print("DAG", DAG)
        # plt.figure(figsize=(8, 4))
        plt.subplot(122)
        plt.title('result' + str(round_id))
        nx.draw(DAG, pos=nx.circular_layout(DAG), with_labels=True)
        plt.savefig("result\\result" + str(round_id) + ".png")

        f = open('result\\result.txt', 'a', encoding='utf-8')  # a 追加模式
        f.write("\n\nSHD" + str(round_id) +": "+ str(SHD(accumulated_mat, adj_matrix.toarray())))
        # f.write("\nSHD" + str(round_id) + str(SHD(accumulated_mat, self.__clients[0]._local_dag_dataset.adj_matrix)))
        f.close()


        print("accumulated_mat: ", accumulated_mat)
        print("torch.from_numpy(accumulated_mat: ",torch.from_numpy(accumulated_mat))
        print("self.global_dataset_dag.adj_matrix: ", self.global_dataset_dag.adj_matrix)
        print("torch.from_numpy(self.global_dataset_dag.adj_matrix): ",torch.from_numpy(self.global_dataset_dag.adj_matrix))
        accumulated_mat_tensor = torch.from_numpy(accumulated_mat)
        adj_matrix_tensor = torch.from_numpy(self.global_dataset_dag.adj_matrix)
        false_positives = torch.logical_and(torch.from_numpy(accumulated_mat).bool(), ~torch.from_numpy(self.global_dataset_dag.adj_matrix).bool())
        false_negatives = torch.logical_and(~torch.from_numpy(accumulated_mat).bool(), torch.from_numpy(self.global_dataset_dag.adj_matrix).bool())
        TP = torch.logical_and(torch.from_numpy(accumulated_mat).bool(), torch.from_numpy(self.global_dataset_dag.adj_matrix).bool()).float().sum().item()
        # TN = torch.logical_and(~accumulated_mat, ~self._local_dag_dataset.adj_matrix).float().sum().item()
        FP = false_positives.float().sum().item()
        FN = false_negatives.float().sum().item()
        # TN = TN - self.gamma.shape[-1]  # Remove diagonal as those are not being predicted
        recall = TP / max(TP + FN, 1e-5)
        precision = TP / max(TP + FP, 1e-5)
        f1_score = 2 * precision * recall / (precision + recall)  # F1
        orient_TP = torch.logical_and(adj_matrix_tensor == 1, accumulated_mat_tensor == 1).float().sum().item()
        orient_FN = torch.logical_and(adj_matrix_tensor == 1, accumulated_mat_tensor == 0).float().sum().item()
        orient_acc = orient_TP / max(1e-5, orient_TP + orient_FN)
        f = open('result\\result.txt', 'a', encoding='utf-8')  # a 追加模式
        f.write("\nTP" + str(round_id) +": "+ str(TP))
        f.write("\nFN" + str(round_id) +": "+ str(FN))
        f.write("\nFP" + str(round_id) +": "+ str(FP))
        f.write("\nf1_score" + str(round_id) +": "+ str(f1_score))
        f.write("\nrecall" + str(round_id) +": "+ str(recall))
        f.write("\nprecision" + str(round_id) +": " + str(precision))
        f.write("\norient_acc" + str(round_id) +": " + str(orient_acc))
        f.write("\norient_TP" + str(round_id) + ":" + str(orient_TP))
        f.write("\norient_FN" + str(round_id) + ":" + str(orient_FN))
        f.close()



        self.prior_gamma = prior_gamma
        self.prior_theta = prior_theta
        return prior_gamma, prior_theta

    def update_results(self, prior_gamma: np.ndarray, prior_theta: np.ndarray):
        """ Update the results dictionary for each round.

        Args:
            prior_gamma (numpy.ndarray): Edge existence matrix at the end of the round.
            prior_theta (numpy.ndarray): Edge orientation matrix acquired at the end of the round.
        """
        self.results['round_gammas'].append(prior_gamma)
        self.results['round_thetas'].append(prior_theta)

        ground_truth_matrix = self.global_dataset_dag.adj_matrix
        round_discovered_matrix = FederatedSimulator.get_binary_adjacency_mat(prior_gamma, prior_theta)
        round_acyclic_matrix = FederatedSimulator.get_acyclic_adjacency_mat(prior_gamma, prior_theta)


        self.results['round_adjs'].append(round_discovered_matrix)
        self.results['round_acycle_adjs'].append(round_acyclic_matrix)

        for client in self.__clients:
            self.results[f'client_{client.get_client_id()}_adjs'].append(client.binary_adjacency_mat)
            self.results[f'client_{client.get_client_id()}_metrics'].append(client.metrics_dict)
            self.results[f'client_{client.get_client_id()}_metrics_acycle'].append(client.metrics_dict_acycle)
        logger.info(f'original matrix: \n {self.global_dataset_dag.adj_matrix}')
        logger.info(f'End of the round results: \n {round_discovered_matrix} \n')

    def save_results(self):
        """ Save the results dictionary as a pickle file.
        """
        file_dir = os.path.join(self.__output_dir, f'results_{self.__experiment_id}_{self.__repeat_id}.pickle')
        with open(file_dir, 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        cache_dir = os.path.join(self.__output_dir, '.mpcache')
        # shutil.rmtree(cache_dir)

    @staticmethod
    def get_binary_adjacency_mat(gamma: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """ Calculate the adjacency matrix based on gamma and theta matrices.

        Args:
            gamma (numpy.ndarray): Edge existence matrix.
            theta (numpy.ndarray): Edge orientation matrix.

        Returns:
            numpy.ndarray: Binary adjacency matrix.
        """

        return (((gamma > 0.0) * (theta > 0.0)) == 1).astype(int)

    @staticmethod
    def get_acyclic_adjacency_mat(gamma: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """ Calculate the adjacency matrix based on acyclicity constraint.

        Args:
            gamma (numpy.ndarray): Edge existence matrix.
            theta (numpy.ndarray): Edge orientation matrix.

        Returns:
            numpy.ndarray: Binary and acyclic adjacency matrix.
        """
        gamma_t = torch.from_numpy(gamma)
        theta_t = torch.from_numpy(theta)

        acycle_mat_tensor = find_best_acyclic_graph(gamma=torch.sigmoid(gamma_t),
                                                    theta=torch.sigmoid(theta_t))

        return acycle_mat_tensor.numpy()

    @staticmethod
    def adjust_theta(prior_theta):
        """ Make the orientation matrix comply with ENCO rule of e_i,j + e_j,i = 1.

        Args:
            prior_theta (numpy.ndarray): Aggregated theta matrix.

        Returns:
            numpy.ndarray: Adjusted edge orientation matrix.
        """

        error_mat = prior_theta + prior_theta.T
        for v_i, v_j in np.transpose(np.nonzero(error_mat)):
            prob = np.max([np.abs(prior_theta[v_i][v_j]), np.abs(prior_theta[v_j][v_i])])
            prior_theta[v_i][v_j] = prob if prior_theta[v_i][v_j] > 0 else -prob
            prior_theta[v_j][v_i] = prob if prior_theta[v_i][v_j] < 0 else -prob

        return prior_theta

if __name__ == '__main__':
    datasets: Dict[str, CausalDAGDataset] = dict()
    files = sorted(glob("../data/asia.bif"))

    for f in files:
        graph = load_graph_file(f)
        print(f, "-> %i nodes, %i categories overall" %
              (graph.num_vars, sum([v.prob_dist.num_categs for v in graph.variables])))

        original_adjacency_mat = graph.adj_matrix
        n_vars = original_adjacency_mat.shape[0]
        logger.debug(f'Global dataset adjacency matrix: \n {original_adjacency_mat.astype(int)}')

        data_obs =graph.sample(batch_size=n_vars * 500, as_array=True)
        logger.info(f'Shape of global observational data: {data_obs.shape}')

        data_int = ENCOAlg.sample_int_data(graph, n_vars * 50)
        # data_int = np.zeros((11, 200, 11))
        logger.info(f'Shape of global interventional data: {data_int.shape}\n')
        print("data_obs:\n", data_obs)
        print("data_int:\n", data_int)
        dataset = CausalDAGDataset(original_adjacency_mat, data_obs, data_int)

        print(f'Shape of Obs data: {dataset.data_obs.shape}')
        print(f'Shape of Int data: {dataset.data_int.shape}')
        print(f'Excluded interventions: {graph.exclude_inters}')

        interventions_dict = {0: [var_idx for var_idx in range(n_vars)], 1: [var_idx for var_idx in range(n_vars)],
                              2: [var_idx for var_idx in range(n_vars)],3: [var_idx for var_idx in range(n_vars)],
                              4: [var_idx for var_idx in range(n_vars)], 5: [var_idx for var_idx in range(n_vars)],
                              6: [var_idx for var_idx in range(n_vars)], 7: [var_idx for var_idx in range(n_vars)],
                              8: [var_idx for var_idx in range(n_vars)], 9: [var_idx for var_idx in range(n_vars)]
                              }

        feature_missing_dict = {0: [0,1,2,3], 1: [4,5,6,7], 2: [0,1,2,3], 3: [4,5,6,7],
                                4: [0,1,2,3], 5: [4,5,6,7], 6: [0,1,2,3], 7: [4,5,6,7], 8:[0,1,2,3],9: [4,5,6,7]}

        num_vars_dict = {0: 4, 1: 4, 2: 4,3:4,4:4,5:4,6:4,7:4,8:4,9:4}
        federated_model = FederatedSimulator(interventions_dict, num_clients=10, num_rounds=10, client_parallelism=False)
        federated_model.initialize_clients_data(external_global_dataset=dataset,feature_missing_dict=feature_missing_dict, num_vars_dict=num_vars_dict)
        federated_model.execute_simulation(aggregation_method="naive_num",
                                           initial_mass=np.array([16, 16]),
                                           alpha=0.2, beta=0.3, min_mass=0.1)