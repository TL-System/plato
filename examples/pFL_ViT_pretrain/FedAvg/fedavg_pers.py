"""
A federated learning server supporting the personalized learning,

The major property of this server to perform the personalized learning
on all participanting clients at the final round.
.
"""

import os
import random
import logging

from plato.servers import fedavg
from plato.config import Config
from plato.utils import fonts


class Server(fedavg.Server):
    """ Federated learning server to support the training of ssl models. """

    def __init__(self,
                 model=None,
                 datasource=None,
                 algorithm=None,
                 trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

        # the clients' id that has been selected to performed
        # the linear evaluation.
        # thus, these clients should be masked in the current
        # selection.
        self.visited_clients_id = []
        # whether to perform the eval at the final round
        self.do_perform_final_eval = False

        self.do_eval_stop_learning = False

    def choose_clients(self, clients_pool, clients_count):
        """ Choose a subset of the clients to participate in each round. """
        assert clients_count <= len(clients_pool)
        random.setstate(self.prng_state)

        # if we decide to perfrom the eval test in the final round
        if hasattr(Config().clients, "do_final_personalization") and Config(
        ).clients.do_final_personalization:

            # the number of clients have not been visited
            non_visited_clients_count = len(clients_pool) - len(
                self.visited_clients_id)

            # if the training reaches the final round
            # performing the client selection for the final
            # round
            if self.current_round >= Config().trainer.rounds:
                # perfrom the eval test by mandotary
                Config().clients = Config().clients._replace(
                    pers_learning_interval=1)
                Config().clients = Config().clients._replace(do_test=True)

                # to perfrom the eval at final round
                self.do_perform_final_eval = True

                if non_visited_clients_count <= clients_count:
                    # if the non visited clients is less than the
                    # required clients per round,
                    # select all left non visited clients
                    # then this is the actually final round
                    # the training should be stopped after this
                    # selection,
                    # therefore, no need to minus current_round by 1
                    clients_count = non_visited_clients_count
                    self.do_eval_stop_learning = True
                    # we must change the clients_per_round to be
                    # the number of clients_count, i.e., how many
                    # clients will be selected in this round.
                    # By doing so, the server can know how many updates
                    # to be received to processing.
                    self.clients_per_round = non_visited_clients_count

                # maintain current round to be the final round
                if self.current_round > Config().trainer.rounds:
                    self.current_round = Config().trainer.rounds

                logging.info(
                    fonts.colourize(
                        f"\n Performing personalization on {clients_count} clients at final round {self.current_round}.",
                        colour='red',
                        style='bold'))
                # remove the visited clients from the clients_pool
                clients_pool = [
                    client_id for client_id in clients_pool
                    if client_id not in self.visited_clients_id
                ]
        else:
            # directly set the do_eval_stop_learning to be true
            # when the user does not want to perform the final eval test
            # by doing so, the stop will be triggered by only the normal
            # case
            self.do_eval_stop_learning = True

        # Select clients randomly
        selected_clients = random.sample(clients_pool, clients_count)

        # add the selected clients id to the visited
        if self.do_perform_final_eval:
            self.visited_clients_id += selected_clients

        self.prng_state = random.getstate()
        logging.info("[%s] Selected clients: %s", self, selected_clients)
        return selected_clients

    def model_strucuture_logging(self):

        # logging the personalzied model's info
        datasources = Config().data.datasource

        whole_model_file_name = f"defined_whole_model.log"
        global_model_file_name = f"global_model.log"
        model_path = Config().params['model_path']
        to_save_dir = model_path
        os.makedirs(to_save_dir, exist_ok=True)
        to_save_path = os.path.join(to_save_dir, whole_model_file_name)
        to_save_global_path = os.path.join(to_save_dir, global_model_file_name)
        if not os.path.exists(to_save_path):
            defined_model_state = str(self.trainer.model)

            with open(to_save_path, 'w') as f:
                f.write(defined_model_state)

            logging.info("Logging the defined whole model' information to %s",
                         to_save_dir)

        if not os.path.exists(to_save_global_path):
            # obtain the global model as ordered dict
            global_model_state_dict = self.algorithm.extract_weights()
            global_parameter_names = list(global_model_state_dict.keys())
            with open(to_save_global_path, 'w') as f:
                for item in global_parameter_names:
                    f.write("%s\n" % item)

            logging.info("Logging the global model' information to %s",
                         to_save_global_path)

    def init_trainer(self):
        super().init_trainer()
        # logging the defined model to dir
        self.model_strucuture_logging()

    async def wrap_up(self):
        """ Wrapping up when each round of training is done.

            This function is required to be extended if
            the user want to support:
                perform the evaluation stage when the
                learning reach the target_accuracy
                or target_perplexity.

            Currently, we only support:
                perform evaluation in the final round.


        """
        self.save_to_checkpoint()

        # Break the loop when the target accuracy is achieved
        # target_accuracy = None
        # target_perplexity = None

        # if hasattr(Config().trainer, 'target_accuracy'):
        #     target_accuracy = Config().trainer.target_accuracy
        # elif hasattr(Config().trainer, 'target_perplexity'):
        #     target_perplexity = Config().trainer.target_perplexity

        # if target_accuracy and self.accuracy >= target_accuracy:
        #     logging.info("[%s] Target accuracy reached.", self)
        #     await self.close()

        # if target_perplexity and self.accuracy <= target_perplexity:
        #     logging.info("[%s] Target perplexity reached.", self)
        #     await self.close()

        if self.current_round >= Config(
        ).trainer.rounds and self.do_eval_stop_learning:
            logging.info("Target number of training rounds reached.")
            await self.close()