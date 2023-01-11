"""
A simple federated learning server capable of selecting clients unseen during
training.

The total clients are divided into two parts, referred to as
1.- participant clients
2.- unparticipant clients

"""

from typing import List
from collections import UserDict
from collections.abc import Iterable
import random
import logging

from plato.servers import fedavg
from plato.config import Config


class TupleAsKeyDict(UserDict):
    def key_status(self, key):
        target_key = [
            data_key
            for data_key in self.data
            if key == data_key or (isinstance(data_key, Iterable) and key in data_key)
        ]

        if target_key:
            return target_key[0], True

        return None, False

    def __getitem__(self, key):

        target_key, is_key_existed = self.key_status(key)

        if is_key_existed:
            return self.data[target_key]

        if hasattr(self.__class__, "__missing__"):
            return self.__class__.__missing__(self, key)
        raise KeyError(key)

    def __contains__(self, key):
        """Whether a key contained in the dict."""
        return self.key_status(key)[-1]


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

        self.personalization_groups = None

        # the flag denoting whether all clients
        # perform personalization based on the
        # received global model.
        # two special value -1 and 0,
        # if -1, do personalization after final round
        # if 0, ignore personalization
        self.do_personalization_interval = 0
        self.do_personalization_group = "Selected"
        self.personalization_status_info = {}
        self.personalization_group_type_info = {}

        # the clients that have completed
        # personalization
        self.personalization_done_clients_pool = []

        # the flag denoting whether the personalization
        # has been started
        self.personalization_started = False
        # the flag denoting whether the personalization
        # has been terminated
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
            loaded_config.clients.participant_clients_ratio
            if hasattr(loaded_config.clients, "participant_clients_ratio")
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
        self.do_personalization_group = (
            loaded_config.server.do_personalization_group
            if hasattr(loaded_config.server, "do_personalization_group")
            else "Selected"
        )

        self.personalization_status_info = TupleAsKeyDict(
            {
                0: "No personalization required.",
                -1: "Personalization after the final round.",
                tuple(range(1, loaded_config.trainer.rounds)): (
                    "Personalization every {} rounds."
                ).format(self.do_personalization_interval),
            }
        )

        self.personalization_group_type_info = {
            "selected": "Personalization on selected clients",
            "total": "Personalization on total clients",
            "participant": "Personalization on participanting clients",
            "nonparticipant": "Personalization on non-participanting clients",
        }
        self.personalization_groups = {
            "selected": None,
            "total": self.clients_pool,
            "participant": self.participant_clients_pool,
            "nonparticipant": self.nonparticipant_clients_pool,
        }

        if self.do_personalization_interval == 0:
            logging.info("[%s] No personalization will be performed.", self)
        elif self.do_personalization_interval == -1:
            logging.info(
                "[%s] Personalization will be performed after final rounds.", self
            )
            logging.info(
                "[%s] %s.",
                self,
                self.personalization_group_type_info[
                    self.do_personalization_group.lower()
                ],
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
                self.personalization_group_type_info[
                    self.do_personalization_group.lower()
                ],
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
            self.personalization_groups["participant"] = self.participant_clients_pool
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
            self.personalization_groups[
                "nonparticipant"
            ] = self.nonparticipant_clients_pool

        if not self.personalization_groups["total"]:
            self.personalization_groups["total"] = self.clients_pool

    def before_clients_sampling(self, clients_pool, clients_count):
        """Determine clients pool and clients count before samling clients."""

        # by default, we run the general federated training
        # the clients pool should be participant clients
        clients_pool = self.participant_clients_pool

        # personalization after final round is required
        if self.do_personalization_interval == -1:

            if self.current_round > Config().trainer.rounds:
                print("clients_pool: ", clients_pool)
                clients_pool = self.personalization_groups[
                self.do_personalization_group.lower()
                ]
                print("pers clients_pool: ", clients_pool)
                print(
                    "self.personalization_done_clients_pool: ",
                    self.personalization_done_clients_pool,
                )
                self.personalization_started = True
                self.current_round = Config().trainer.rounds

                # the number of clients have not been visited
                non_visited_clients_count = len(clients_pool) - len(
                    self.personalization_done_clients_pool
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

                # remove the visited clients from the clients_pool
                clients_pool = [
                    client_id
                    for client_id in clients_pool
                    if client_id not in self.personalization_done_clients_pool
                ]

        else:
            # directly set the do_eval_stop_learning to be true
            # when the user does not want to perform the final eval test
            # by doing so, the stop will be triggered by only the normal
            # case
            self.personalization_terminated = False

        return clients_pool, clients_count

    def after_clients_sampling(self, selected_clients, **kwargs):
        """Perform operations after clients sampling."""
        # personalization after final round is required
        if self.personalization_started:
            self.personalization_done_clients_pool += selected_clients

    def choose_clients(self, clients_pool, clients_count):
        """Chooses a subset of the clients to participate in each round."""

        # obtain the clients participanting in federated training
        # the user need to know that the clients_pool holds the id
        # of all available clients by default
        self.set_participant_clients_pool(clients_pool)

        clients_pool, clients_count = self.before_clients_sampling(
            clients_pool, clients_count
        )

        assert clients_count <= len(self.participant_clients_pool)

        random.setstate(self.prng_state)

        # Select clients randomly
        selected_clients = random.sample(clients_pool, clients_count)

        self.after_clients_sampling(selected_clients)

        self.prng_state = random.getstate()
        if selected_clients == len(clients_pool):
            logging.info("[%s] Selected all %d clients", self, len(selected_clients))
        else:
            logging.info("[%s] Selected clients: %s", self, selected_clients)

        return selected_clients

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
                logging.info(
                    "%s Completed.",
                    self.personalization_status_info[self.do_personalization_interval],
                )

                await self._close()

