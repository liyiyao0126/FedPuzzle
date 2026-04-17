
import logging
import os, sys
import pickle

import networkx as nx
import torch
import numpy as np
import random
from typing import Dict, List
import psutil
import time
import json
from memory_profiler import profile
import tracemalloc
import argparse
from matplotlib import pyplot as plt

sys.path.append("../")
from federated.utils import find_shortest_distance_dict
from federated.logging_settings import logger
from federated.causal_learning import ENCOAlg
from Enco.causal_graphs.graph_definition import CausalDAGDataset
from Enco.causal_discovery.utils import find_best_acyclic_graph

from cdt.metrics import SHD
def generate_feature_missing_dict(num_clients, num_vars, missing_per_client):
    var_covered = [False] * num_vars
    feature_missing_dict = {}

    var_to_client = [random.randint(0, num_clients - 1) for _ in range(num_vars)]

    for client_id in range(num_clients):
        feature_missing_dict[client_id] = set(range(num_vars))

    for var, client in enumerate(var_to_client):
        feature_missing_dict[client].discard(var)

    for client_id in range(num_clients):
        current_missing = feature_missing_dict[client_id]
        if len(current_missing) > missing_per_client:
            keep_vars = set(random.sample(list(current_missing), missing_per_client))
            feature_missing_dict[client_id] = keep_vars
        elif len(current_missing) < missing_per_client:
            possible_vars = set(range(num_vars)) - current_missing
            need = missing_per_client - len(current_missing)
            add_vars = set(random.sample(list(possible_vars), need))
            feature_missing_dict[client_id] = current_missing | add_vars

    feature_missing_dict = {k: sorted(list(v)) for k, v in feature_missing_dict.items()}
    return feature_missing_dict

def monitor_resources(func):

    def wrapper(*args, **kwargs):
        tracemalloc.start()
        process = psutil.Process()

        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"{func.__name__} - Execution Time: {end_time - start_time:.2f}s")
        print(f"{func.__name__} - Memory Usage: {end_memory - start_memory:.2f}MB")
        print(f"{func.__name__} - Peak Memory: {peak / 1024 / 1024:.2f}MB")

        return result

    return wrapper


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
        self.memory_usage_log = []
        self.time_log = []

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
    @monitor_resources
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
    @monitor_resources
    def execute_simulation(self, aggregation_method: str = "naive", num_epochs: int = 2, gamma_threshold: float = 0.5,theta_threshold: float = 0.5,delta_threshold: float = 0.0001,
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
            process = psutil.Process()
            memory_info = process.memory_info()
            self.memory_usage_log.append({
                'round': round_id,
                'rss': memory_info.rss / 1024 / 1024,  # MB
                'vms': memory_info.vms / 1024 / 1024,  # MB
                'time': time.time()
            })
            """ Inference stage"""
            self.infer_local_models(prior_gamma, prior_theta, num_epochs, round_id,gamma_threshold,theta_threshold,delta_threshold)

            """ Aggregation stage """
            agg_gamma, agg_theta = self.aggregate_clients_updates(aggregation_method, round_id,gamma_threshold,theta_threshold,delta_threshold, **kwargs)

            """ Store round results """
            self.update_results(agg_gamma, agg_theta,gamma_threshold,theta_threshold)

            """ Incorporate beliefs"""

            prior_gamma, prior_theta = agg_gamma, agg_theta

        """ Save the final results """
        self.save_results()

        logger.info(f'Finishing experiment {self.__experiment_id}\n')
    @monitor_resources
    def infer_local_models(self, prior_gamma: np.ndarray, prior_theta: np.ndarray, num_epochs, round_id,gamma_threshold,theta_threshold,delta_threshold):
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

                client.infer_causal_structure(round_id,gamma_threshold,theta_threshold,delta_threshold, prior_gamma1, prior_theta1, num_epochs)

    def aggregate_clients_updates(self, aggregation_method, round_id,gamma_threshold,theta_threshold,delta_threshold, **kwargs):
        """Perform aggregation step for all clients.

        Args:
            aggregation_method (str): Can be naive or locality aggregation so far.
            round_id (int): Current roung id.

        Returns:
            np.ndarray, np.ndarray: Aggregated gamma and theta matrices.
        """

        if aggregation_method == "naive":
            agg_gamma, agg_theta = self.naive_aggregation(round_id=round_id,gamma_threshold=gamma_threshold,theta_threshold=theta_threshold,delta_threshold=delta_threshold)
        if aggregation_method == "naive_num":
            agg_gamma, agg_theta = self.naive_aggregation_num(round_id=round_id)

        return agg_gamma, agg_theta

    def insert_column(self, A, j, new_column):
        left = A[:, :j]
        right = A[:, j:]
        shape = (A.shape[0], A.shape[1] + 1)
        result = np.zeros(shape, dtype=A.dtype)
        result[:, :j] = left
        result[:, j] = new_column.flatten()
        result[:, j + 1:] = right
        return result

    def insert_row(self, A, i, new_row):
        top = A[:i, :]
        bottom = A[i:, :]
        return np.vstack((top, new_row, bottom))

    def get_row_minus_column(self, matrix, i, j):
        row = matrix[i, :]
        row_without_jth_element = np.delete(row, j)
        return row_without_jth_element

    def calculate_local_metrics(self, round_id, gamma_threshold, theta_threshold):

        local_skeleton_metrics = []
        local_csg_metrics = []

        global_true_adj = self.global_dataset_dag.adj_matrix.copy()
        client_details = []
        for client in self.__clients:
            client_id = client.get_client_id()
            missing_vars = self.feature_missing_dict[client_id]

            inferred_existence_mat = client.inferred_existence_mat
            inferred_orientation_mat = client.inferred_orientation_mat

            prior_gamma_t = torch.from_numpy(inferred_existence_mat)
            prior_theta_t = torch.from_numpy(inferred_orientation_mat)
            binary_matrix_gamma = (torch.sigmoid(prior_gamma_t) > gamma_threshold).numpy().astype(int)
            binary_matrix_theta = (torch.sigmoid(prior_theta_t) > theta_threshold).numpy().astype(int)
            pred_adj_matrix = binary_matrix_gamma * binary_matrix_theta

            client_true_adj = np.copy(global_true_adj)
            if len(missing_vars) > 0:
                client_true_adj = np.delete(client_true_adj, missing_vars, axis=0)
                client_true_adj = np.delete(client_true_adj, missing_vars, axis=1)

            true_skeleton = np.maximum(client_true_adj, client_true_adj.T)
            pred_skeleton = np.maximum(pred_adj_matrix, pred_adj_matrix.T)

            np.fill_diagonal(true_skeleton, 0)
            np.fill_diagonal(pred_skeleton, 0)

            TP = np.sum((true_skeleton == 1) & (pred_skeleton == 1))
            FP = np.sum((true_skeleton == 0) & (pred_skeleton == 1))
            FN = np.sum((true_skeleton == 1) & (pred_skeleton == 0))

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            true_graph = nx.from_numpy_array(true_skeleton)
            pred_graph = nx.from_numpy_array(pred_skeleton)
            shd = SHD(nx.adjacency_matrix(true_graph).toarray(), nx.adjacency_matrix(pred_graph).toarray())

            local_skeleton_metrics.append({
                'SHD': shd,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            client_star_graph_info = {
                'client_id': client_id,
                'star_graphs': []
            }
            if hasattr(client, 'adjset') and hasattr(client, '_local_dag_dataset'):
                client_true_adj_local = client._local_dag_dataset.adj_matrix
                for i in range(len(client.adjset)):
                    if i < client_true_adj_local.shape[0]:
                        true_parents = np.where(client_true_adj_local[:, i] > 0)[0]
                        true_children = np.where(client_true_adj_local[i, :] > 0)[0]

                        true_neighbors = \
                            np.where((client_true_adj_local[:, i] > 0) | (client_true_adj_local[i, :] > 0))[0]

                        pred_neighbors = client.adjset[i] if len(client.adjset) > i else np.array([])

                        if hasattr(client, 'csgset') and len(client.csgset) > i:
                            pred_csg = client.csgset[i] if len(client.csgset[i]) > 0 else np.array([])
                            pred_parents = set()
                            pred_children = set()
                            for idx, neighbor in enumerate(pred_neighbors):
                                if idx < len(pred_csg):
                                    if pred_csg[idx] == 1:
                                        pred_parents.add(neighbor)
                                    else:
                                        pred_children.add(neighbor)
                        else:
                            pred_parents = set()
                            pred_children = set(pred_neighbors)

                        direction_errors = 0
                        total_true_edges = 0

                        for neighbor in true_neighbors:
                            total_true_edges += 1

                            if neighbor in true_parents:
                                if neighbor not in pred_parents:
                                    direction_errors += 1
                            elif neighbor in true_children:
                                if neighbor not in pred_children:
                                    direction_errors += 1

                        direction_accuracy = (
                                                         total_true_edges - direction_errors) / total_true_edges if total_true_edges > 0 else 0

                        star_graph_info = {
                            'node_index': i,
                            'true_parents': true_parents.tolist(),
                            'true_children': true_children.tolist(),
                            'pred_parents': list(pred_parents),
                            'pred_children': list(pred_children),
                            'true_neighbors': true_neighbors.tolist(),
                            'pred_neighbors': pred_neighbors.tolist(),
                            'direction_errors': direction_errors,
                            'total_true_edges': total_true_edges,
                            'direction_accuracy': direction_accuracy
                        }
                        client_star_graph_info['star_graphs'].append(star_graph_info)

                        local_csg_metrics.append({
                            'SHD': direction_errors,
                            'precision': direction_accuracy,
                            'recall': direction_accuracy,
                            'f1': direction_accuracy
                        })

            client_details.append(client_star_graph_info)

            missing_per_client = len(self.feature_missing_dict[0]) if self.feature_missing_dict else 0
            result_filename = f"result\\local_metrics_round{round_id}_missing{missing_per_client}.txt"
            with open(result_filename, 'a', encoding='utf-8') as f:
                f.write(f"Round {round_id} - Local Metrics\n")
                f.write("=" * 40 + "\n")

                if local_skeleton_metrics:
                    avg_skeleton_shd = np.mean([m['SHD'] for m in local_skeleton_metrics])
                    avg_skeleton_precision = np.mean([m['precision'] for m in local_skeleton_metrics])
                    avg_skeleton_recall = np.mean([m['recall'] for m in local_skeleton_metrics])
                    avg_skeleton_f1 = np.mean([m['f1'] for m in local_skeleton_metrics])

                    f.write("Local Skeleton Metrics:\n")
                    f.write(f"  Average SHD: {avg_skeleton_shd:.2f}\n")
                    f.write(f"  Average Precision: {avg_skeleton_precision:.4f}\n")
                    f.write(f"  Average Recall: {avg_skeleton_recall:.4f}\n")
                    f.write(f"  Average F1: {avg_skeleton_f1:.4f}\n")
                    f.write("\n")

                    print(f"Round {round_id} - Average Local Skeleton Metrics:")
                    print(f"  SHD: {avg_skeleton_shd:.2f}")
                    print(f"  Precision: {avg_skeleton_precision:.4f}")
                    print(f"  Recall: {avg_skeleton_recall:.4f}")
                    print(f"  F1: {avg_skeleton_f1:.4f}")

                if local_csg_metrics:
                    avg_csg_shd = np.mean([m['SHD'] for m in local_csg_metrics])
                    avg_csg_precision = np.mean([m['precision'] for m in local_csg_metrics])
                    avg_csg_recall = np.mean([m['recall'] for m in local_csg_metrics])
                    avg_csg_f1 = np.mean([m['f1'] for m in local_csg_metrics])

                    f.write("Star Graph Metrics (Direction Accuracy Only):\n")
                    f.write(f"  Average SHD: {avg_csg_shd:.2f}\n")
                    f.write(f"  Average Direction Accuracy: {avg_csg_precision:.4f}\n")

                    print(f"Round {round_id} - Average Star Graph Metrics:")
                    print(f"  Direction Accuracy: {avg_csg_precision:.4f}")


    @monitor_resources
    def naive_aggregation(self, round_id,gamma_threshold,theta_threshold,delta_threshold):
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
                inferred_existence_mat = self.insert_row(inferred_existence_mat,
                                                         self.feature_missing_dict[client.get_client_id()][i],
                                                         self.get_row_minus_column(self.prior_gamma,
                                                                                   self.feature_missing_dict[
                                                                                       client.get_client_id()][i],
                                                                                   self.feature_missing_dict[
                                                                                       client.get_client_id()]))
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_existence_mat = self.insert_column(inferred_existence_mat,
                                                            self.feature_missing_dict[client.get_client_id()][i],
                                                            self.prior_gamma[:,
                                                            self.feature_missing_dict[client.get_client_id()][i]])
            weighted_gamma_mat = inferred_existence_mat * client.get_accessible_percentage()
            accumulated_gamma_mat += weighted_gamma_mat
            #
            inferred_orientation_mat = client.inferred_orientation_mat
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_orientation_mat = self.insert_row(inferred_orientation_mat,
                                                           self.feature_missing_dict[client.get_client_id()][i],
                                                           self.get_row_minus_column(self.prior_theta,
                                                                                     self.feature_missing_dict[
                                                                                         client.get_client_id()][i],
                                                                                     self.feature_missing_dict[
                                                                                         client.get_client_id()]))
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_orientation_mat = self.insert_column(inferred_orientation_mat,
                                                              self.feature_missing_dict[client.get_client_id()][i],
                                                              self.prior_theta[:,
                                                              self.feature_missing_dict[client.get_client_id()][i]])
            weighted_theta_mat = inferred_orientation_mat * client.get_accessible_percentage()
            accumulated_theta_mat += weighted_theta_mat
            weights += client.get_accessible_percentage()
            if len(self.feature_missing_dict[client.get_client_id()]) != 0:
                for feature_missing_var in self.feature_missing_dict[client.get_client_id()]:
                    weights[feature_missing_var] -= 100
            csgset.append(client.csgset)
            adjset.append(client.adjset)
        accumulated_mat = np.zeros_like(accumulated_theta_mat)
        prior_gamma: np.ndarray = accumulated_gamma_mat / weights
        prior_theta: np.ndarray = accumulated_theta_mat / weights
        prior_gamma_t = torch.from_numpy(prior_gamma)
        prior_theta_t = torch.from_numpy(prior_theta)
        binary_matrix_gamma = (torch.sigmoid(prior_gamma_t) > gamma_threshold).numpy().astype(int)
        binary_matrix_theta = (torch.sigmoid(prior_theta_t) > theta_threshold).numpy().astype(int)
        binary_matrix = binary_matrix_gamma * binary_matrix_theta
        undirect = np.maximum(binary_matrix, binary_matrix.T)
        for client in self.__clients:
            for i in range(len(adjset[client.get_client_id()])):
                for j in range(len(adjset[client.get_client_id()][i])):
                    for k in range(len(self.feature_missing_dict[client.get_client_id()])):

                        if adjset[client.get_client_id()][i][j] > self.feature_missing_dict[client.get_client_id()][k]:
                            adjset[client.get_client_id()][i][j] += 1

        for client in self.__clients:
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                csgset[client.get_client_id()].insert(self.feature_missing_dict[client.get_client_id()][i],
                                                      np.array([]))
                adjset[client.get_client_id()].insert(self.feature_missing_dict[client.get_client_id()][i],
                                                      np.array([]))
        for i in range(len(undirect)):
            for j in range(len(undirect[i])):
                if undirect[i][j] > 0:
                    k = 0
                    count = 0
                    length = 0
                    for client in self.__clients:

                        if j in client.adjset[i]:
                            count = count + 1
                            k = k + client.csgset[i][np.where(client.adjset[i] == j)[0][0]] * len(client.adjset[i])
                            length = length + len(client.adjset[i])
                        if i in client.adjset[j]:
                            count = count + 1
                            k = k - (client.csgset[j][np.where(client.adjset[j] == i)[0][0]] + 1) * len(
                                client.adjset[j])
                            length = length - len(client.adjset[j])

                    if k > 0:
                        accumulated_mat[i][j] = 1
                        accumulated_mat[j][i] = 0
                    else:
                        accumulated_mat[i][j] = 0
                        accumulated_mat[j][i] = 1

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

        DAG = nx.from_numpy_array(accumulated_mat, create_using=nx.DiGraph())
        plt.subplot(122)
        plt.title('result' + str(round_id))
        nx.draw(DAG, pos=nx.circular_layout(DAG), with_labels=True)
        plt.savefig("result\\result" + str(round_id) + ".png")
        result_filename = f"result\\result_gamma{gamma_threshold}_theta{theta_threshold}_delta{delta_threshold}.txt"
        print(result_filename)
        f = open(result_filename, 'a', encoding='utf-8')

        f.write("\n\nSHD" + str(round_id) + ": " + str(SHD(accumulated_mat, adj_matrix.toarray())))
        f.close()

        accumulated_mat_tensor = torch.from_numpy(accumulated_mat)
        adj_matrix_tensor = torch.from_numpy(self.global_dataset_dag.adj_matrix)
        false_positives = torch.logical_and(torch.from_numpy(accumulated_mat).bool(),
                                            ~torch.from_numpy(self.global_dataset_dag.adj_matrix).bool())
        false_negatives = torch.logical_and(~torch.from_numpy(accumulated_mat).bool(),
                                            torch.from_numpy(self.global_dataset_dag.adj_matrix).bool())
        TP = torch.logical_and(torch.from_numpy(accumulated_mat).bool(),
                               torch.from_numpy(self.global_dataset_dag.adj_matrix).bool()).float().sum().item()
        # TN = torch.logical_and(~accumulated_mat, ~self._local_dag_dataset.adj_matrix).float().sum().item()
        FP = false_positives.float().sum().item()
        FN = false_negatives.float().sum().item()
        # TN = TN - self.gamma.shape[-1]  # Remove diagonal as those are not being predicted
        recall = TP / max(TP + FN, 1e-5)
        precision = TP / max(TP + FP, 1e-5)
        if (precision + recall) == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)  # F1
        orient_TP = torch.logical_and(adj_matrix_tensor == 1, accumulated_mat_tensor == 1).float().sum().item()
        orient_FN = torch.logical_and(adj_matrix_tensor == 1, accumulated_mat_tensor == 0).float().sum().item()
        orient_acc = orient_TP / max(1e-5, orient_TP + orient_FN)
        f = open(result_filename, 'a', encoding='utf-8')
        f.write("\nTP" + str(round_id) + ": " + str(TP))
        f.write("\nFN" + str(round_id) + ": " + str(FN))
        f.write("\nFP" + str(round_id) + ": " + str(FP))
        f.write("\nf1_score" + str(round_id) + ": " + str(f1_score))
        f.write("\nrecall" + str(round_id) + ": " + str(recall))
        f.write("\nprecision" + str(round_id) + ": " + str(precision))
        f.write("\norient_acc" + str(round_id) + ": " + str(orient_acc))
        f.write("\norient_TP" + str(round_id) + ":" + str(orient_TP))
        f.write("\norient_FN" + str(round_id) + ":" + str(orient_FN))
        f.close()
        self.calculate_local_metrics(round_id, gamma_threshold, theta_threshold)

        self.prior_gamma = prior_gamma
        self.prior_theta = prior_theta
        return prior_gamma, prior_theta


    @monitor_resources
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
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_existence_mat = self.insert_row(inferred_existence_mat,
                                                         self.feature_missing_dict[client.get_client_id()][i],
                                                         self.get_row_minus_column(self.prior_gamma,
                                                                                   self.feature_missing_dict[
                                                                                       client.get_client_id()][i],
                                                                                   self.feature_missing_dict[
                                                                                       client.get_client_id()]))
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_existence_mat = self.insert_column(inferred_existence_mat,
                                                            self.feature_missing_dict[client.get_client_id()][i],
                                                            self.prior_gamma[:,
                                                            self.feature_missing_dict[client.get_client_id()][i]])
            weighted_gamma_mat = inferred_existence_mat * client.get_accessible_percentage()
            accumulated_gamma_mat += weighted_gamma_mat

            inferred_orientation_mat = client.inferred_orientation_mat
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_orientation_mat = self.insert_row(inferred_orientation_mat,
                                                           self.feature_missing_dict[client.get_client_id()][i],
                                                           self.get_row_minus_column(self.prior_theta,
                                                                                     self.feature_missing_dict[
                                                                                         client.get_client_id()][i],
                                                                                     self.feature_missing_dict[
                                                                                         client.get_client_id()]))
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                inferred_orientation_mat = self.insert_column(inferred_orientation_mat,
                                                              self.feature_missing_dict[client.get_client_id()][i],
                                                              self.prior_theta[:,
                                                              self.feature_missing_dict[client.get_client_id()][i]])
            weighted_theta_mat = inferred_orientation_mat * client.get_accessible_percentage()
            accumulated_theta_mat += weighted_theta_mat

            weights += client.get_accessible_percentage()

            csgset.append(client.csgset)
            adjset.append(client.adjset)
        accumulated_mat = np.zeros_like(accumulated_theta_mat)
        prior_gamma: np.ndarray = accumulated_gamma_mat / weights
        prior_theta: np.ndarray = accumulated_theta_mat / weights
        binary_matrix_gamma = (prior_gamma > 0).astype(int)
        binary_matrix_theta = (prior_theta > 0).astype(int)
        binary_matrix = binary_matrix_gamma * binary_matrix_theta
        undirect = np.maximum(binary_matrix, binary_matrix.T)
        for client in self.__clients:
            for i in range(len(adjset[client.get_client_id()])):
                for j in range(len(adjset[client.get_client_id()][i])):
                    for k in range(len(self.feature_missing_dict[client.get_client_id()])):

                        if adjset[client.get_client_id()][i][j] > self.feature_missing_dict[client.get_client_id()][k]:
                            adjset[client.get_client_id()][i][j] += 1

        for client in self.__clients:
            for i in range(len(self.feature_missing_dict[client.get_client_id()])):
                csgset[client.get_client_id()].insert(self.feature_missing_dict[client.get_client_id()][i],
                                                      np.array([]))
                adjset[client.get_client_id()].insert(self.feature_missing_dict[client.get_client_id()][i],
                                                      np.array([]))
        for i in range(len(undirect)):
            for j in range(len(undirect[i])):
                if accumulated_gamma_mat[i][j] > 0:
                    k = 0
                    count = 0
                    for client in self.__clients:

                        if j in client.adjset[i]:
                            count = count + 1
                            k = k + client.csgset[i][np.where(client.adjset[i] == j)[0][0]]
                        if i in client.adjset[j]:
                            count = count + 1
                            k = k - client.csgset[j][np.where(client.adjset[j] == i)[0][0]] + 1

                    if k > count / 2:
                        accumulated_mat[i][j] = 1
                        accumulated_mat[j][i] = 0
                    else:
                        accumulated_mat[i][j] = 0
                        accumulated_mat[j][i] = 1

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

        DAG = nx.from_numpy_array(accumulated_mat, create_using=nx.DiGraph())
        plt.subplot(122)
        plt.title('result' + str(round_id))
        nx.draw(DAG, pos=nx.circular_layout(DAG), with_labels=True)
        plt.savefig("result\\result" + str(round_id) + ".png")

        f = open('result\\result.txt', 'a', encoding='utf-8')  # a 追加模式
        f.write("\n\nSHD" + str(round_id) + ": " + str(SHD(accumulated_mat, adj_matrix.toarray())))
        f.close()

        accumulated_mat_tensor = torch.from_numpy(accumulated_mat)
        adj_matrix_tensor = torch.from_numpy(self.global_dataset_dag.adj_matrix)
        false_positives = torch.logical_and(torch.from_numpy(accumulated_mat).bool(),
                                            ~torch.from_numpy(self.global_dataset_dag.adj_matrix).bool())
        false_negatives = torch.logical_and(~torch.from_numpy(accumulated_mat).bool(),
                                            torch.from_numpy(self.global_dataset_dag.adj_matrix).bool())
        TP = torch.logical_and(torch.from_numpy(accumulated_mat).bool(),
                               torch.from_numpy(self.global_dataset_dag.adj_matrix).bool()).float().sum().item()
        FP = false_positives.float().sum().item()
        FN = false_negatives.float().sum().item()
        recall = TP / max(TP + FN, 1e-5)
        precision = TP / max(TP + FP, 1e-5)
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)  # F1

        orient_TP = torch.logical_and(adj_matrix_tensor == 1, accumulated_mat_tensor == 1).float().sum().item()
        orient_FN = torch.logical_and(adj_matrix_tensor == 1, accumulated_mat_tensor == 0).float().sum().item()
        orient_acc = orient_TP / max(1e-5, orient_TP + orient_FN)
        f = open('result\\result.txt', 'a', encoding='utf-8')
        f.write("\nTP" + str(round_id) + ": " + str(TP))
        f.write("\nFN" + str(round_id) + ": " + str(FN))
        f.write("\nFP" + str(round_id) + ": " + str(FP))
        f.write("\nf1_score" + str(round_id) + ": " + str(f1_score))
        f.write("\nrecall" + str(round_id) + ": " + str(recall))
        f.write("\nprecision" + str(round_id) + ": " + str(precision))
        f.write("\norient_acc" + str(round_id) + ": " + str(orient_acc))
        f.write("\norient_TP" + str(round_id) + ":" + str(orient_TP))
        f.write("\norient_FN" + str(round_id) + ":" + str(orient_FN))
        f.close()

        self.prior_gamma = prior_gamma
        self.prior_theta = prior_theta
        return prior_gamma, prior_theta


    def plot_memory_usage(self):
        if not self.memory_usage_log:
            print("No memory usage data available")
            return

        rounds = [log['round'] for log in self.memory_usage_log]
        rss_values = [log['rss'] for log in self.memory_usage_log]
        vms_values = [log['vms'] for log in self.memory_usage_log]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(rounds, rss_values, marker='o', label='RSS (Resident Set Size)')
        plt.xlabel('Federated Round')
        plt.ylabel('Memory Usage (MB)')
        plt.title('RSS Memory Usage Over Rounds')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(rounds, vms_values, marker='s', label='VMS (Virtual Memory Size)')
        plt.xlabel('Federated Round')
        plt.ylabel('Memory Usage (MB)')
        plt.title('VMS Memory Usage Over Rounds')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plot_filename = f'memory_usage_clients{self.__num_clients}_vars{self.__num_vars}.png'
        plt.savefig(os.path.join(self.__output_dir, plot_filename), dpi=300, bbox_inches='tight')
        plt.close()

    def save_resource_logs(self):

        memory_filename = f'memory_usage_clients{self.__num_clients}_vars{self.__num_vars}.json'
        with open(os.path.join(self.__output_dir, memory_filename), 'w') as f:
            json.dump(self.memory_usage_log, f, indent=2)

        if self.time_log:
            time_filename = f'time_log_clients{self.__num_clients}_vars{self.__num_vars}.json'
            with open(os.path.join(self.__output_dir, time_filename), 'w') as f:
                json.dump(self.time_log, f, indent=2)

    def save_results(self):
        """ Save the results dictionary as a pickle file.
        """
        file_dir = os.path.join(self.__output_dir, f'results_{self.__experiment_id}_{self.__repeat_id}.pickle')
        with open(file_dir, 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.plot_memory_usage()
        self.save_resource_logs()

        cache_dir = os.path.join(self.__output_dir, '.mpcache')

    def update_results(self, prior_gamma: np.ndarray, prior_theta: np.ndarray, gamma_threshold, theta_threshold):
        """ Update the results dictionary for each round.

        Args:
            prior_gamma (numpy.ndarray): Edge existence matrix at the end of the round.
            prior_theta (numpy.ndarray): Edge orientation matrix acquired at the end of the round.
        """

        self.results['round_gammas'].append(prior_gamma)
        self.results['round_thetas'].append(prior_theta)

        ground_truth_matrix = self.__clients[0].original_adjacency_mat
        round_discovered_matrix = FederatedSimulator.get_binary_adjacency_mat(prior_gamma, prior_theta,gamma_threshold,theta_threshold)
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

    @staticmethod
    def get_binary_adjacency_mat(gamma: np.ndarray, theta: np.ndarray, gamma_threshold, theta_threshold) -> np.ndarray:
        gamma_t = torch.from_numpy(gamma)
        theta_t = torch.from_numpy(theta)
        return (((torch.sigmoid(gamma_t) > gamma_threshold) * (
                    torch.sigmoid(theta_t) > theta_threshold)) == 1).numpy().astype(int)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=5, help='客户端数量')
    parser.add_argument('--num_vars', type=int, default=10, help='变量数量')
    parser.add_argument('--missing_per_client', type=int, default=5, help='每个客户端缺失的变量数')
    parser.add_argument('--gamma_threshold', type=float, default=0.5, help='边缘概率门限')
    parser.add_argument('--theta_threshold', type=float, default=0.5, help='边缘方向概率门限')
    parser.add_argument('--delta_threshold', type=float, default=0.05, help='1')

    args = parser.parse_args()
    num_clients = args.num_clients
    num_vars = args.num_vars
    missing_per_client = args.missing_per_client
    interventions_dict = {i: list(range(num_vars)) for i in range(num_clients)}
    num_vars_dict = {i: num_vars-missing_per_client for i in range(num_clients)}
    feature_missing_dict = generate_feature_missing_dict(num_clients, num_vars, missing_per_client)
    print(feature_missing_dict)
    all_missing = [set(v) for v in feature_missing_dict.values()]
    for var in range(num_vars):
        if all(var in missing for missing in all_missing):
            exit(1)
    federated_model = FederatedSimulator(interventions_dict, num_clients=num_clients, num_rounds=10, client_parallelism=False)
    federated_model.initialize_clients_data(num_vars=num_vars, graph_type="random", edge_prob=0.3,
                                            feature_missing_dict=feature_missing_dict, num_vars_dict=num_vars_dict)
    federated_model.execute_simulation(aggregation_method="naive",
                                       initial_mass=np.array([16, 16]),
                                       alpha=0.2, beta=0.3, min_mass=0.1, gamma_threshold=args.gamma_threshold,theta_threshold=args.theta_threshold, delta_threshold=args.delta_threshold)
