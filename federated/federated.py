
import logging
import os, sys
import pickle

import networkx as nx
import torch
import numpy as np

from typing import Dict, List

from matplotlib import pyplot as plt

sys.path.append("../")
from federated.utils import find_shortest_distance_dict
from federated.logging_settings import logger
from federated.causal_learning import ENCOAlg
from client.causal_graphs.graph_definition import CausalDAGDataset
from client.causal_discovery.utils import find_best_acyclic_graph

from cdt.metrics import SHD


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
                prior_gamma1 = prior_gamma
                prior_theta1 = prior_theta
                if round_id != 0:
                    prior_gamma1 = np.delete(prior_gamma1, self.feature_missing_dict[client.get_client_id()], axis=0)
                    prior_gamma1 = np.delete(prior_gamma1, self.feature_missing_dict[client.get_client_id()], axis=1)
                    prior_theta1 = np.delete(prior_theta1, self.feature_missing_dict[client.get_client_id()], axis=0)
                    prior_theta1 = np.delete(prior_theta1, self.feature_missing_dict[client.get_client_id()], axis=1)

                client.infer_causal_structure(round_id, prior_gamma1, prior_theta1, num_epochs)

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
         """ The corresponding code will be made publicly available upon publication of this article."""

    def naive_aggregation_num(self, round_id):
        """ The corresponding code will be made publicly available upon publication of this article."""

    def update_results(self, prior_gamma: np.ndarray, prior_theta: np.ndarray):
        """ Update the results dictionary for each round.

        Args:
            prior_gamma (numpy.ndarray): Edge existence matrix at the end of the round.
            prior_theta (numpy.ndarray): Edge orientation matrix acquired at the end of the round.
        """

        self.results['round_gammas'].append(prior_gamma)
        self.results['round_thetas'].append(prior_theta)

        ground_truth_matrix = self.__clients[0].original_adjacency_mat
        round_discovered_matrix = FederatedSimulator.get_binary_adjacency_mat(prior_gamma, prior_theta)
        round_acyclic_matrix = FederatedSimulator.get_acyclic_adjacency_mat(prior_gamma, prior_theta)


        self.results['round_adjs'].append(round_discovered_matrix)
        # self.results['round_metrics'].append(round_metrics)
        self.results['round_acycle_adjs'].append(round_acyclic_matrix)
        # self.results['round_acycle_metrics'].append(round_acycle_metrics)

        for client in self.__clients:
            self.results[f'client_{client.get_client_id()}_adjs'].append(client.binary_adjacency_mat)
            self.results[f'client_{client.get_client_id()}_metrics'].append(client.metrics_dict)
            self.results[f'client_{client.get_client_id()}_metrics_acycle'].append(client.metrics_dict_acycle)

        logger.info(f'End of the round results: \n {round_acyclic_matrix} \n')
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

    interventions_dict = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                          2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                          4: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
    feature_missing_dict = {0: [0], 1: [1, 2], 2: [3, 4, 5], 3: [4, 5, 6, 7], 4: [5, 6, 7, 8, 9]}
    num_vars_dict = {0: 9, 1: 8, 2: 7, 3: 6, 4: 5}
    federated_model = FederatedSimulator(interventions_dict, num_clients=5, num_rounds=10, client_parallelism=True)
    federated_model.initialize_clients_data(num_vars=10, graph_type="random", edge_prob=0.3,
                                            feature_missing_dict=feature_missing_dict, num_vars_dict=num_vars_dict)
    federated_model.execute_simulation(aggregation_method="naive",
                                       initial_mass=np.array([16, 16]),
                                       alpha=0.2, beta=0.3, min_mass=0.1)
