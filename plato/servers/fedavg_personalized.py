"""
A simple federated learning server capable of selecting clients unseen during
training.

The total clients are divided into two parts, referred to as
1.- participant clients
2.- unparticipant clients

"""

from typing import List
import random
import logging

from plato.servers import fedavg
from plato.config import Config


class Server(fedavg.Server):
    """Federated learning server controling the client selection."""

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        # two types of client groups
        # utilized clients during federated training
        self.participant_clients = 0
        # unused clients during federated training
        self.nonparticipant_clients = 0

        # percentage of participating customers among all customers
        # by default, all clients will participant
        self.participant_clients_ratio = 1.0

        # clients id of two types of clients
        self.participant_clients_pool = None
        self.nonparticipant_clients_pool = None

        # the flag denoting whether all clients
        # perform personalization based on the
        # received global model.
        # two special value -1 and 0,
        # if -1, do personalization after final round
        # if 0, ignore personalization
        self.do_personalization_interval = 0
        self.do_personalization_type = "Selected"

        # the clients that have been visited
        # this is generally utilized in the
        # final personalization stage
        self.visited_clients_pool = []

        # the flag denoting whether the personalization
        # has been terminated
        # this will not be set unless the
        # personalization setup is opened
        self.personalization_terminated = False

        self.initialize_personalization()
        self.check_hyper_parameters()

    def check_hyper_parameters(self):
        """Check whether the hyper-parameters are set correctly."""
        # `the participant_clients_pool` and `participant_clients_ratio`
        # if these two hyper-parameters are set simutaneously
        #  they should correspond to each other.

        if hasattr(Config().clients, "participant_clients_ratio") and hasattr(
            Config().clients, "participant_clients_pool"
        ):
            assert self.participant_clients == len(self.participant_clients_pool)

        assert (
            self.total_clients == self.participant_clients + self.nonparticipant_clients
        )

    def initialize_personalization(self):
        """Initialize two types of clients."""
        # set participant and nonparticipant clients
        loaded_config = Config()

        ## 1. initialize participanting clients
        self.participant_clients_ratio = (
            loaded_config.server.participant_clients_ratio
            if hasattr(loaded_config.server, "participant_clients_ratio")
            else 1.0
        )

        #  clients will / will not participant in federated training
        self.participant_clients = int(
            self.total_clients * self.participant_clients_ratio
        )
        self.nonparticipant_clients = int(self.total_clients - self.participant_clients)
        logging.info(
            "[%s] Total clients (%d), participanting clients (%d) ratio (%.3f), nonparticipant_clients (%d)",
            self,
            self.total_clients,
            self.participant_clients,
            self.participant_clients_ratio,
            self.nonparticipant_clients,
        )

        ## 2. initialize personalization interval and types
        # if the value is set to be -1, thus the personalization
        # will be perform at the end of full communication rounds.
        # if do_personalization_interval is 0, no personalization
        # will be performed
        # by default, this value is set to be 0.
        self.do_personalization_interval = (
            loaded_config.server.do_personalization_interval
            if hasattr(loaded_config.server, "do_personalization_interval")
            else 0
        )
        self.do_personalization_type = (
            loaded_config.server.do_personalization_type
            if hasattr(loaded_config.server, "do_personalization_type")
            else "Selected"
        )

        personalization_type_info = {
            "selected": "Personalization on selected clients",
            "total": "Personalization on total clients",
            "participant": "Personalization on participanting clients",
            "nonparticipant": "Personalization on non-participanting clients",
        }

        if self.do_personalization_interval == 0:
            logging.info("[%s] No personalization will be performed.", self)
        elif self.do_personalization_interval == -1:
            logging.info(
                "[%s] Personalization will be performed after final rounds.", self
            )
            self.do_personalization_type = "Total"
            logging.info(
                "[%s] %s.",
                self,
                personalization_type_info[self.do_personalization_type.lower()],
            )
        else:
            logging.info(
                "[%s] Personalization will be performed every %d rounds.",
                self,
                self.do_personalization_interval,
            )
            logging.info(
                "[%s] %s.",
                self,
                personalization_type_info[self.do_personalization_type.lower()],
            )

    def set_participant_clients_pool(self, clients_pool: List[int]):
        """Set the participant clients id.

        :param clients_pool: A list holding the id of all possible
            clients.

        Note, the participant clients pool will be set once without
        any modification afterwards.
        """
        loaded_config = Config()

        assert self.participant_clients <= len(clients_pool)

        # only set the clients pool when they are empty.

        # load the participant clients pool from configuration
        # if this is not provided,
        # The first `self.participant_clients` in clients_pool will
        # be utilized as participant clients.
        if self.participant_clients_pool is None:
            self.participant_clients_pool = (
                loaded_config.clients.participant_clients_id
                if hasattr(loaded_config.clients, "participant_clients_id")
                else clients_pool[: self.participant_clients]
            )
            logging.info(
                "[%s] Prepared participanting clients pool: %s",
                self,
                self.participant_clients_pool,
            )
        # load the non-participant clients pool from configuration
        # if this is not provided,
        # Then reminding clients of clients_pool apart from the
        # participant_clients will be utilized as non-participant clients.
        if self.nonparticipant_clients_pool is None:
            self.nonparticipant_clients_pool = (
                loaded_config.clients.nonparticipant_clients_pool
                if hasattr(loaded_config.clients, "nonparticipant_clients_id")
                else [
                    client_id
                    for client_id in clients_pool
                    if client_id not in self.participant_clients_pool
                ]
            )
            logging.info(
                "[%s] Prepared non-participanting clients pool: %s",
                self,
                self.nonparticipant_clients_pool,
            )

    def determine_clients_pool(self, clients_pool, clients_count):
        """Determine clients pool and clients count."""

        # if final round and personalization is required
        if self.do_personalization_interval == -1:
            if self.current_round >= Config().trainer.rounds:

                # the number of clients have not been visited
                non_visited_clients_count = len(clients_pool) - len(
                    self.visited_clients_pool
                )
                if non_visited_clients_count <= clients_count:
                    # if the non visited clients is less than the
                    # required clients per round,
                    # select all left non visited clients
                    # then personalization on all clients has
                    # been terminated
                    clients_count = non_visited_clients_count

                    # we must change the clients_per_round to be
                    # the number of clients_count, i.e., how many
                    # clients will be selected in this round.
                    # By doing so, the server can know how many updates
                    # to be received for aggregation.
                    self.clients_per_round = non_visited_clients_count

                    # personalization has been terminated
                    self.personalization_terminated = True

                # maintain current round to be the final round
                if self.current_round > Config().trainer.rounds:
                    self.current_round = Config().trainer.rounds

                # remove the visited clients from the clients_pool
                clients_pool = [
                    client_id
                    for client_id in clients_pool
                    if client_id not in self.visited_clients_pool
                ]

            # select all possible clients
            clients_pool = clients_pool
            # perform personalization on all clients, including
            # participant and non-participant clients
            clients_count = self.participant_clients + self.nonparticipant_clients
        else:
            # directly set the do_eval_stop_learning to be true
            # when the user does not want to perform the final eval test
            # by doing so, the stop will be triggered by only the normal
            # case
            self.personalization_terminated = False

        return clients_pool, clients_count

    async def wrap_up(self):
        """Wrapping up when each round of training is done.

        This function is required to be extended if
        the user want to support:
            perform the evaluation stage when the
            learning reach the target_accuracy
            or target_perplexity.

        """

        self.save_to_checkpoint()

        if self.current_round >= Config().trainer.rounds:
            logging.info("Target number of training rounds reached.")
            if self.do_personalization_interval == 0 or self.personalization_terminated:
                logging.info("Final Personalization Finished.")
            await self.close()

    def choose_clients(self, clients_pool, clients_count):
        """Chooses a subset of the clients to participate in each round."""

        # obtain the clients participanting in federated training
        # the user need to know that the clients_pool holds the id
        # of all available clients by default
        self.set_participant_clients_pool(clients_pool)

        clients_pool = self.participant_clients_pool

        assert clients_count <= len(self.participant_clients_pool)

        random.setstate(self.prng_state)

        # Select clients randomly
        selected_clients = random.sample(clients_pool, clients_count)

        self.prng_state = random.getstate()
        if selected_clients == len(clients_pool):
            logging.info("[%s] Selected all %d clients", self, len(selected_clients))
        else:
            logging.info("[%s] Selected clients: %s", self, selected_clients)

        return selected_clients
