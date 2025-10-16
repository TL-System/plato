"""
A customized server for Knot, a clustered aggregation mechanism designed for
federated unlearning.
"""

import copy
import logging
import os
import random
from collections import deque

import fedunlearning_server
import numpy
import solver
import torch
import torch.nn.functional as F
from cvxopt import matrix
from numpy.linalg import norm

from plato.config import Config
from plato.utils import fonts


class Server(fedunlearning_server.Server):
    """
    A federated unlearning server that implements the federated unlearning clustering algorithm.

    The work pipeline of the server is as follows:

    1. The server divides the total number of clients randomly into a number of clusters.
    2. The clients carry out the training normally.
    3. Model updates from the clients will be aggregated according to the clustering assignments.
    4. Send the model corresponding to each cluster to its corresponding clients.
    5. Perform global aggregation (aggregate from all clusters) after convergence or when the
       target round has reached.
    """

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
        )

        # A dictionary that maps client IDs to the cluster IDs
        self.num_clusters = 0
        self.clusters = {}

        # If HuggingFace models are used, set the initial accuracy to a very large value
        if hasattr(Config().trainer, "target_perplexity"):
            self.accuracy = 1000

        # A dictionary that maps the cluster ID to a boolean value that shows whether
        # this cluster is currently going through the retraining phase
        self.clustered_retraining = {}

        # A dictionary that maps cluster IDs to the accuracy belonging to each cluster
        # This accuracy is computed based on the test dataset on the server side, i.e.,
        # each cluster will test its updated model after aggregation) on the server's
        # test set to obtain its accuracy.
        self.clustered_test_accuracy = {}

        # The recent history of global accuracies
        self.recent_global_accuracies = None
        self.recent_history_size = None

        # A dictionary that maps the cluster ID to the earliest round that retraining
        # must start from when entering the retraining phase
        self.earliest_round = {}

        # A dictionary that maps the cluster ID to the round when the retraining process
        # should roll back to, which is the earliest_round above minus 1
        self.rollback_round = {}

        # A dictionary that maps cluster IDs to the updates belonging to each cluster
        self.clustered_updates = {}

        # A pre-trained server model generated before the unlearning, for cos similarity
        self.pretrained_server_model = None

        # A dictionary that maps client ids and its cos similarity compared with
        # pre-trained server model
        self.clients_similarity = {}

        # Whether we are using random clustering or optimized clustering
        self.initialize_optimization = False

        # Initialize basic metrics required by clusters, such as accuracites
        self._init_cluster_states()

        # Initialize the function that clustering clients by solver or randomly
        self._clustering_clients()

    def init_trainer(self) -> None:
        """Load the trainer and initialize the dictionary that maps cluster IDs to client IDs."""
        super().init_trainer()

        self.algorithm.init_clusters(self.clusters)

    def choose_clients(self, clients_pool, clients_count):
        """
        Choose a subset of clients to participate in each round.
        When do_optimized_clustering is true, the first and second round is to
        extract training time and cos similarity for all clients;
        after and at second round, training process resume.
        """
        assert clients_count <= len(clients_pool)

        if (
            hasattr(Config().server, "do_optimized_clustering")
            and Config().server.do_optimized_clustering
        ):
            if self.current_round <= 2 and self.initialize_optimization:
                clients_count = len(clients_pool)
                self.minimum_clients = clients_count
            elif self.current_round == 3:
                self.minimum_clients = Config.server.minimum_clients_aggregated
                self.clients_per_round = Config().clients.per_round
                clients_count = self.clients_per_round

        selected_clients = self._select_clients_with_strategy(
            clients_pool, clients_count
        )

        logging.info("[%s] Selected clients: %s", self, selected_clients)
        return selected_clients

    def clients_selected(self, selected_clients):
        """Remembers the first round that a particular client ID was selected."""
        if (
            hasattr(Config().server, "do_optimized_clustering")
            and Config().server.do_optimized_clustering
            and not self.initialize_optimization
        ):
            for client_id in selected_clients:
                if client_id not in self.round_first_selected:
                    self.round_first_selected[client_id] = self.current_round

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """
        Aggregate the reported weight updates from the selected clients,
        according to each client's clustering assignment, using the
        federated averaging algorithm.

        Only clients belonging to the same cluster will be aggregated.
        """
        if (
            self.current_round == 1
            and (
                hasattr(Config().server, "do_optimized_clustering")
                and Config().server.do_optimized_clustering
            )
            and self.initialize_optimization
        ):
            # Perform normal server aggregation by first computing weight deltas
            deltas_received = self.algorithm.compute_weight_deltas(
                baseline_weights, weights_received, cluster_id=None
            )
            # and then run a framework-agnostic server aggregation algorithm, such as
            # the federated averaging algorithm
            logging.info("[%s] Aggregating model weight deltas.", self)
            deltas = await self.aggregate_deltas(self.updates, deltas_received)
            # Updates the existing model weights from the provided deltas
            updated_weights = self.algorithm.update_weights(deltas)
            return updated_weights

        if (
            self.current_round == 2
            and hasattr(Config().server, "do_optimized_clustering")
            and Config().server.do_optimized_clustering
            and self.initialize_optimization
        ):
            # Compute client clustering using optimization
            self._optimize_clustering(updates)
            return baseline_weights

        self.clustered_updates = {}
        data_deletion_round = Config().clients.data_deletion_round

        for client_update in updates:
            client_id = client_update.client_id
            cluster_id = self.clusters[client_id]

            include_update = True
            if self.clustered_retraining[cluster_id]:
                include_update = (
                    abs(client_update.staleness)
                    <= self.current_round - data_deletion_round - 1
                )

            if not include_update:
                continue

            self.clustered_updates.setdefault(cluster_id, []).append(client_update)

        for cluster_id, cluster_updates in self.clustered_updates.items():
            if not cluster_updates:
                continue

            weights_received = [update.payload for update in cluster_updates]
            deltas_received = self.algorithm.compute_weight_deltas(
                baseline_weights,
                weights_received,
                cluster_id=cluster_id,
            )
            deltas = await self.aggregate_deltas(cluster_updates, deltas_received)
            updated_weights = self.algorithm.update_weights(
                deltas, cluster_id=cluster_id
            )

            self.algorithm.load_weights(updated_weights, cluster_id=cluster_id)

        return baseline_weights

    def weights_aggregated(self, updates):
        """Method called after the updated weights have been aggregated."""
        # Testing the updated clustered model directly at the server
        if (
            hasattr(Config().server, "do_clustered_test")
            and Config().server.do_clustered_test
            and not self.initialize_optimization
        ):
            # First, obtain the set of cluster IDs that have been aggregated
            updated_cluster_ids = {
                self.clusters[update.client_id] for update in self.updates
            }

            test_accuracy_per_cluster = (
                self.trainer.testing_strategy.test_clustered_models(
                    self.testset,
                    self.testset_sampler,
                    self.trainer.context,
                    self.algorithm.models,
                    updated_cluster_ids,
                )
            )

            # Second, update the test accuracy for clusters that have just been tested
            self.clustered_test_accuracy.update(test_accuracy_per_cluster)

        if hasattr(Config().server, "do_test") and Config().server.do_test:
            # Retrieve the model from the cluster with the highest accuracy
            self.trainer.model.load_state_dict(self._aggregate_models(), strict=True)

    def clients_processed(self):
        """Determining the rollback round and roll back to that round, if retraining is needed
        for each of the clusters."""
        # If data_deletion_round equals to the current round at server for the first time,
        # and the clients requesting retraining has been selected before, the retraining
        # phase starts.

        # Make sure that the model at round 0 is loaded at round 3, if we
        # have activated the optimized clustering algorithm
        if (
            hasattr(Config().server, "do_optimized_clustering")
            and Config().server.do_optimized_clustering
            and self.current_round == 2
        ):
            for cluster_id in range(self.num_clusters):
                # Loading the saved model on the server for starting the retraining phase
                checkpoint_path = Config.params["checkpoint_path"]

                model_name = (
                    Config().trainer.model_name
                    if hasattr(Config().trainer, "model_name")
                    else "custom"
                )

                # Loading the model from round 0
                rollback_round = 0
                filename = f"checkpoint_{model_name}_{rollback_round}_{cluster_id}.pth"

                self._load_model(cluster_id, filename, checkpoint_path)

                logging.info(
                    fonts.colourize(
                        f"[Server #{os.getpid()}] Cluster #{cluster_id}: Round #0 "
                        f"model reloaded from {filename}.",
                        colour="green",
                    )
                )

        if not self.initialize_optimization:
            self.earliest_round = {
                cluster_id: self.current_round
                for cluster_id in range(self.num_clusters)
            }

        data_deletion_round = Config().clients.data_deletion_round
        clients_to_delete = Config().clients.clients_requesting_deletion

        for cluster_id in range(self.num_clusters):
            if (
                (self.current_round == data_deletion_round)
                and not self.clustered_retraining[cluster_id]
                and not self.initialize_optimization
            ):
                for client_id, init_round in self.round_first_selected.items():
                    if (
                        client_id in clients_to_delete
                        and self.clusters[client_id] == cluster_id
                    ):
                        self.clustered_retraining[cluster_id] = True

                        if self.earliest_round[cluster_id] > init_round:
                            self.earliest_round[cluster_id] = init_round

                        self.rollback_round[cluster_id] = (
                            self.earliest_round[cluster_id] - 1
                        )
                        logging.info(
                            fonts.colourize(
                                f"[{self}] Data deleted. Retraining cluster #{cluster_id} "
                                f"from the states after round #{self.rollback_round[cluster_id]}.",
                                colour="green",
                            )
                        )

                if cluster_id in self.rollback_round:
                    # Loading the saved model on the server for starting the retraining phase
                    checkpoint_path = Config.params["checkpoint_path"]

                    model_name = (
                        Config().trainer.model_name
                        if hasattr(Config().trainer, "model_name")
                        else "custom"
                    )

                    # When the current_round matches the data deletion round, load the model from
                    # `rollback_round`.
                    rollback_round = self.rollback_round[cluster_id]
                    filename = (
                        f"checkpoint_{model_name}_{rollback_round}_{cluster_id}.pth"
                    )

                    self._load_model(cluster_id, filename, checkpoint_path)

                    logging.info(
                        "[Server #%d] Model in cluster #%s's retraining phase loaded from %s.",
                        os.getpid(),
                        cluster_id,
                        filename,
                    )

        if self.current_round >= 2:
            self.initialize_optimization = False

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """Returns a customrized server response with any additional information."""
        server_response = super().customize_server_response(
            server_response, client_id=client_id
        )

        cluster_id = self.clusters[client_id]

        if (
            cluster_id in self.clustered_retraining
            and self.clustered_retraining[cluster_id]
        ):
            # Each cluster has its own round number that it needs to roll back to during the
            # retraining phase, which is the earliest round among clients in this cluster, minus 1
            server_response["rollback_round"] = self.rollback_round[cluster_id]

        return server_response

    def _select_clients_with_strategy(self, clients_pool, clients_count):
        """Delegate client selection to the configured strategy and update PRNG state."""
        self.context.current_round = self.current_round
        self.context.state["prng_state"] = self.prng_state

        selected_clients = self.client_selection_strategy.select_clients(
            clients_pool, clients_count, self.context
        )

        self.prng_state = self.context.state["prng_state"]
        self.client_selection_strategy.on_clients_selected(
            selected_clients, self.context
        )

        return selected_clients

    def get_logged_items(self) -> dict:
        """Get items to be logged by the LogProgressCallback class in a .csv file."""
        logged_items = super().get_logged_items()

        clusters_accuracy = [
            self.clustered_test_accuracy[cluster_id]
            for cluster_id in range(self.num_clusters)
        ]

        clusters_accuracy = "; ".join([str(acc) for acc in clusters_accuracy])

        logged_items["clusters_accuracy"] = clusters_accuracy

        return logged_items

    def save_to_checkpoint(self) -> None:
        """Save a checkpoint for resuming the training session."""
        super().save_to_checkpoint()

        for cluster_id in range(self.num_clusters):
            checkpoint_path = Config.params["checkpoint_path"]

            model_name = (
                Config().trainer.model_name
                if hasattr(Config().trainer, "model_name")
                else "custom"
            )
            filename = f"checkpoint_{model_name}_{self.current_round}_{cluster_id}.pth"
            logging.info(
                "[%s] Saving the checkpoint for cluster #%s to %s/%s.",
                self,
                cluster_id,
                checkpoint_path,
                filename,
            )
            self._save_model(cluster_id, filename, checkpoint_path)

    async def wrap_up(self) -> None:
        """Wrapping up when each round of training is done."""
        self.save_to_checkpoint()

        # Break the loop when the target accuracy is achieved
        target_accuracy = None
        target_perplexity = None

        if (
            self._did_stablize()
            or self.current_round >= Config().trainer.rounds
            and max(self.clustered_retraining.values())
        ):
            logging.info(
                fonts.colourize(
                    f"[{self}] Terminating at round {self.current_round} out of "
                    f"{Config().trainer.rounds}.",
                    colour="green",
                )
            )

            if self.current_round >= Config().trainer.rounds:
                logging.info("Target number of training rounds reached.")
                await self._close()

            if hasattr(Config().trainer, "target_accuracy"):
                target_accuracy = Config().trainer.target_accuracy
            elif hasattr(Config().trainer, "target_perplexity"):
                target_perplexity = Config().trainer.target_perplexity

            if target_accuracy and self.accuracy >= target_accuracy:
                logging.info(
                    "[%s] Target accuracy and standard deviation reached.", self
                )
                await self._close()

            if target_perplexity and self.accuracy <= target_perplexity:
                logging.info(
                    "[%s] Target perplexity and standard deviation reached.",
                    self,
                )
                await self._close()

    def _load_model(self, cluster_id, filename=None, location=None):
        """Loading pre-trained model weights before retraining from a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.pth"

        logging.info("[Server #%d] Loading a model from %s.", os.getpid(), model_path)

        if cluster_id in self.algorithm.models:
            model = self.algorithm.models[cluster_id]
        else:
            model = self.trainer.model

        model.load_state_dict(torch.load(model_path), strict=True)

    def _save_model(self, cluster_id, filename=None, location=None):
        """Saving the model to a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.pth"

        if cluster_id in self.algorithm.models:
            model = self.algorithm.models[cluster_id]
        else:
            model = self.trainer.model
        torch.save(model.state_dict(), model_path)

        logging.info("[Server #%d] Model saved to %s.", os.getpid(), model_path)

    def _init_cluster_states(self):
        """Initialize the basic dictionaries of the clusters and clients."""
        self.num_clusters = Config().server.clusters

        if hasattr(Config().server, "do_optimized_clustering"):
            self.initialize_optimization = Config().server.do_optimized_clustering
        else:
            self.initialize_optimization = False

        # Initializing the dictionary of boolean values, one per cluster, to record whether
        # we have already entered the retraining phase
        self.clustered_retraining = {
            cluster_id: False for cluster_id in range(self.num_clusters)
        }

        # Initializing the dictionary of float numbers, one per cluster, to record the
        # test accuracy of each cluster
        self.clustered_test_accuracy = {
            cluster_id: 0 for cluster_id in range(self.num_clusters)
        }

        # Initializing the dictionary of cos similarity between pre-trained model and
        # clients' first time updates
        self.clients_similarity = {
            client_id: None for client_id in range(1, self.total_clients + 1)
        }

        # recent_history_size denotes the length of accuracy recording
        # Default: each cluster only records latest 5 test accuracy
        if hasattr(Config().server, "window_size"):
            self.recent_history_size = Config().server.window_size
        else:
            self.recent_history_size = 5

        # Initializing a record of global test accuracies
        self.recent_global_accuracies = deque([], maxlen=self.recent_history_size)

    def _clustering_clients(self):
        """
        Randomly divide clients by their client ids into several clusters, and aggregation
        in clusters. As the baseline of optimization clustering.
        """

        total_clients = Config().clients.total_clients
        self.num_clusters = Config().server.clusters

        # The client IDs range from 1 to total_clients, and they are to be distributed to
        # the clusters, ranging from 0 to (num_clusters - 1)
        random.seed(1)
        if (
            hasattr(Config().server, "do_optimized_clustering")
            and Config().server.do_optimized_clustering
        ):
            # initial self.clusters dictionaty
            for client_id in range(1, total_clients + 1):
                self.clusters[client_id] = None
        else:
            # randomly cluster
            for client_id in range(1, total_clients + 1):
                cluster_id = int(random.random() * self.num_clusters)
                self.clusters[client_id] = cluster_id
            print("clusters: ", self.clusters)

    def _aggregate_models(self):
        """Aggregate the models from the clusters by using the one with the
        highest test accuracy.
        """
        avg_model = {
            name: self.trainer.zeros(weights.shape).to(Config().device())
            for name, weights in self.trainer.model.state_dict().items()
        }

        if hasattr(Config().trainer, "target_perplexity"):
            best_cluster_acc = 1000
        else:
            best_cluster_acc = 0

        best_cluster_id = 0

        for cluster_id, cluster_acc in self.clustered_test_accuracy.items():
            if hasattr(Config().trainer, "target_accuracy"):
                if cluster_acc > best_cluster_acc:
                    best_cluster_acc = cluster_acc
                    best_cluster_id = cluster_id

            elif hasattr(Config().trainer, "target_perplexity"):
                if 0 < cluster_acc < best_cluster_acc:
                    best_cluster_acc = cluster_acc
                    best_cluster_id = cluster_id

        for cluster_id, clustered_model in self.algorithm.models.items():
            if cluster_id == best_cluster_id:
                for name, weights in clustered_model.state_dict().items():
                    avg_model[name] = weights

        return avg_model

    def _did_stablize(self):
        """
        Whether the training process should be terminated based on:
            - The global accuracy corresponds to the highest per-cluster test accuracy.
            - The standard deviation of highest per-cluster test accuracies has
              reached a particular target.
        """

        # We first need to exceed the target accuracy for all the clusters
        if hasattr(Config().trainer, "target_accuracy"):
            target_accuracy = Config().trainer.target_accuracy

            # Update the histories of recent global test accuracies
            global_acc = max(self.clustered_test_accuracy.values())
            self.recent_global_accuracies.append(global_acc)

            # We then need to compute the standard deviation of latest
            # self.recent_history_size global accuracies, and make sure that it
            # is lower than the target_accuracy_std
            standard_deviation = numpy.std(list(self.recent_global_accuracies))
            target_accuracy_std = Config().trainer.target_accuracy_std
            logging.info(
                "[%s] Standard deviation of recent global accuracies: %.2f.",
                self,
                standard_deviation,
            )

            if len(self.recent_global_accuracies) >= self.recent_history_size:
                return (
                    global_acc > target_accuracy
                    and standard_deviation < target_accuracy_std
                )
            else:
                return False

        if hasattr(Config().trainer, "target_perplexity"):
            target_perplexity = Config().trainer.target_perplexity

            global_acc = min(self.clustered_test_accuracy.values())
            # Update the histories of recent global test accuracies
            self.recent_global_accuracies.append(global_acc)

            # We then need to compute the standard deviation of latest
            # self.recent_history_size global accuracies, and make sure that it
            # is lower than the target_accuracy_std
            standard_deviation = numpy.std(list(self.recent_global_accuracies))

            target_perplexity_std = Config().trainer.target_perplexity_std

            if len(self.recent_global_accuracies) >= self.recent_history_size:
                return (
                    0 < global_acc < target_perplexity
                    and standard_deviation < target_perplexity_std
                )
            else:
                return False

    def _extract_training_times(self, updates):
        """Extract the training time from the report in client updates."""
        # Initialize a dictionary that maps client_ids to its training times
        client_training_times = {
            client_id: 0 for client_id in range(1, Config().clients.total_clients + 1)
        }

        for update in updates:
            if client_training_times[update.client_id] != 0:
                continue

            client_training_times[update.client_id] = update.report.training_time

        return client_training_times

    def _cosine_similarity(self, updates):
        """Compute the cosine similarity of the received updates and the difference
        between the first round model - initial model and clients' updates."""

        model_name = (
            Config().trainer.model_name
            if hasattr(Config().trainer, "model_name")
            else "custom"
        )
        filename = f"checkpoint_{model_name}_{self.current_round - 2}.pth"
        checkpoint_path = Config.params["checkpoint_path"]
        model_path = f"{checkpoint_path}/{filename}"

        initial_model = copy.deepcopy(self.trainer.model)
        initial_model.load_state_dict(torch.load(model_path))

        initial = torch.zeros(0)
        for __, weight in initial_model.cpu().state_dict().items():
            initial = torch.cat((initial, weight.view(-1)))

        current = torch.zeros(0)
        for __, weight in self.trainer.model.cpu().state_dict().items():
            current = torch.cat((current, weight.view(-1)))

        weights_received = [update.payload for update in self.updates]
        baseline_weights = self.algorithm.extract_weights()
        deltas_received = self.algorithm.compute_weight_deltas(
            baseline_weights, weights_received, cluster_id=None
        )

        for i, update in enumerate(deltas_received):
            client_id = updates[i].client_id
            if self.clients_similarity[client_id] is not None:
                continue
            deltas = torch.zeros(0)
            for __, delta in update.items():
                deltas = torch.cat((deltas, delta.view(-1)))

            similarity = (F.cosine_similarity(current - initial, deltas, dim=0) + 1) / 2
            self.clients_similarity[client_id] = similarity.item()

    def _convert_to_solver(self, client_training_times):
        """Transform useful dictionaries to solvable matrix."""
        # Transfer the values of dic to list.
        training_time_list = list(client_training_times.values())
        similarity_list = list(self.clients_similarity.values())

        # Anchor intervals between clusters: (max value - min value) / the number of clusters
        similarity_interval = (
            max(similarity_list) - min(similarity_list)
        ) / self.num_clusters

        training_time_interval = (
            max(training_time_list) - min(training_time_list)
        ) / self.num_clusters

        # Produce matrices containing the distances between input data and cluster anchors
        training_anchor_distances = []
        similarity_anchor_distances = []

        for cluster_id in range(self.num_clusters):
            training_anchor_distances.append(
                [
                    abs(
                        training_time_interval * cluster_id
                        + min(training_time_list)
                        - time
                    )
                    for time in training_time_list
                ]
            )
            similarity_anchor_distances.append(
                [
                    abs(
                        similarity_interval * cluster_id
                        + min(similarity_list)
                        - similarity
                    )
                    for similarity in similarity_list
                ]
            )

        training_time_np_matrix = numpy.matrix(training_anchor_distances)
        similarity_np_matrix = numpy.matrix(similarity_anchor_distances)

        # Use scaler to tune the weight of training time and similarity on opt problem.
        # The range of original training time is 0 - 30, similarity is 0 - 1.
        # If training_time_scaler = 1, and similarity_scaler = 10,
        # the weights of training time and similarity are both 1/2.
        training_time_scaler = 1
        similarity_scaler = 1

        training_time_matrix = training_time_np_matrix * training_time_scaler
        similarity_matrix = similarity_np_matrix * similarity_scaler

        # Transfer numpy.matrix to cvxopt.matrix for indexing.
        training_time_matrix = matrix(training_time_matrix)
        similarity_matrix = matrix(similarity_matrix)

        l2_list = []

        for client_id in self.clusters:
            for cluster_id in range(self.num_clusters):
                # The key of self.clusters are client_ids, the index start from 1
                # Thus, we use (client_id - 1) as the matrix index here.
                l2_array = numpy.array(
                    [
                        training_time_matrix[cluster_id, client_id - 1],
                        similarity_matrix[cluster_id, client_id - 1],
                    ]
                )

                # Calculate the norm.
                l2_norm = norm(l2_array, 2)
                l2_list.append(l2_norm)

        l2_matrix = matrix(l2_list, training_time_np_matrix.shape)

        return l2_matrix

    def _optimize_clustering(self, updates):
        """
        Computing optimized clustering with an optimization solver,
        based on both per-client training times and the cosine similarity,
        computed between the clients and the global anchors.
        """
        logging.info(
            fonts.colourize(
                f"\n[{self}] Starting the solve the optimization problem to assign "
                f"{len(self.clusters)} clients to {self.num_clusters} clusters.",
                colour="green",
            )
        )

        client_training_times = self._extract_training_times(updates)
        self._cosine_similarity(updates)

        assignment_list = solver.solve(
            # workload_max
            # (len(self.clusters) / self.num_clusters) * 2,
            # len(self.clusters) / self.num_clusters,
            len(self.clusters) / 2,
            # workload_min
            # len(self.clusters) / (self.num_clusters * 2),
            # len(self.clusters) / self.num_clusters,
            len(self.clusters) / self.num_clusters,
            1,
            self.num_clusters,
            len(self.clusters),
            self._convert_to_solver(client_training_times),
        )

        self._convert_from_solver(assignment_list)

    def _convert_from_solver(self, assignment_list):
        """Convert the assignment that generate from solver to clients_clusters dictionary."""
        assignment_array = numpy.array(assignment_list)

        index_of_value_1 = numpy.array(numpy.argwhere(assignment_array == 1))

        self.clusters = dict(zip(index_of_value_1[:, 1] + 1, index_of_value_1[:, 0]))
        self.algorithm.init_clusters(self.clusters)

        logging.info(
            fonts.colourize(
                f"\n[{self}] Cluster assignments: {self.clusters}",
                colour="green",
            )
        )
