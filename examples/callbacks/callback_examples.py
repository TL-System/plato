"""Callback examples  for test purpose"""

import logging

from plato.callbacks.client import ClientCallback
from plato.callbacks.server import ServerCallback
from plato.callbacks.trainer import TrainerCallback


class argumentClientCallback(ClientCallback):
    def on_inbound_received(self, client, inbound_processor):
        logging.info(f"[{client}] Client callback from argument.")


class dynamicClientCallback(ClientCallback):
    def on_inbound_received(self, client, inbound_processor):
        logging.info(f"[{client}] Client callback from dynamic adding.")


class argumentServerCallback(ServerCallback):
    def on_weights_received(self, server, weights_received):
        logging.info(f"[{server}] Server callback from argument.")


class dynamicServerCallback(ServerCallback):
    def on_weights_received(self, server, weights_received):
        logging.info(f"[{server}] Server callback from dynamic adding.")


class customTrainerCallback(TrainerCallback):
    def on_train_run_start(self, trainer, config):
        logging.info(
            f"[Client {trainer.client_id}] Trainer callback from dynamic adding."
        )
