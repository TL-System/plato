"""
Implements additive secret sharing algorithm to encrypt model parameters.
"""
import copy
import pickle
import random
import logging
from typing import Any

from kazoo.client import KazooClient
from kazoo.recipe.lock import Lock

from plato.processors import model
from plato.config import Config
from plato.utils import s3


class Processor(model.Processor):
    """
    A processor that decrypts model tensors
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client_share_index = 0
        self.s3_client = None
        self.client_id = kwargs["client_id"]
        self.zk = None
        self.lock = kwargs["file_lock"]

    # Randomly split a tensor into num_shares
    def split_tensor(self, tensor, num_shares):
        if num_shares == 1:
            tensors = [tensor]
            return tensors

        # First split evenly
        tensors = [tensor / num_shares for i in range(num_shares)]

        # Generate a random number for each share
        start = -0.5
        end = 0.5
        rand_nums = [random.uniform(start, end) for i in range(num_shares - 1)]
        rand_nums.append(0 - sum(rand_nums))

        # Add the random numbers to secret shares
        for i in range(num_shares):
            tensors[i] += rand_nums[i]

        return tensors

    def process(self, data: Any) -> Any:
        # Load round_info object
        if hasattr(Config().server, "s3_endpoint_url"):
            self.zk = KazooClient(
                hosts=f"{Config().server.zk_address}:{Config().server.zk_port}"
            )
            self.zk.start()
            lock = Lock(self.zk, "/my/lock/path")
            lock.acquire()
            logging.info("[%s] Acquired Zookeeper lock", self)

            self.s3_client = s3.S3()
            s3_key = "round_info"
            logging.debug("Retrieving round_info from S3")
            round_info = self.s3_client.receive_from_s3(s3_key)
        else:
            round_info_filename = "mpc_data/round_info"
            self.lock.acquire()
            with open(round_info_filename, "rb") as round_info_file:
                round_info = pickle.load(round_info_file)

        num_clients = len(round_info["selected_clients"])

        # Store the client's weights before encryption in a file for testing
        weights_filename = (
            f"mpc_data/raw_weights_round{round_info['round_number']}"
            f"_client{self.client_id}"
        )
        file = open(weights_filename, "w", encoding="utf8")
        file.write(str(data))
        file.close()

        # Split weights randomly into n shares
        # Initialize data_shares to the shape of data
        data_shares = [copy.deepcopy(data) for i in range(num_clients)]

        # Iterate over the keys of data to split
        for key in data.keys():
            # multiply by num_samples used to train the client
            data[key] *= round_info[f"client_{self.client_id}_info"]["num_samples"]

            # Split tensor randomly into num_clients shares
            tensor_shares = self.split_tensor(data[key], num_clients)

            # Store tensor_shares into data_shares for the particular key
            for i in range(num_clients):
                data_shares[i][key] = tensor_shares[i]

        # Store secret shares in round_info
        for i, client_id in enumerate(round_info["selected_clients"]):
            # Skip the client itself
            if client_id == self.client_id:
                self.client_share_index = (
                    i  # keep track of the index to return the client's share in the end
                )
                continue

            if round_info[f"client_{client_id}_info"]["data"] is None:
                round_info[f"client_{client_id}_info"]["data"] = data_shares[i]
            else:
                for key in data.keys():
                    round_info[f"client_{client_id}_info"]["data"][key] += data_shares[
                        i
                    ][key]

        logging.debug("Print round_info keys before filling client data")
        logging.debug(round_info.keys())

        # Store round_info object
        if hasattr(Config().server, "s3_endpoint_url"):
            self.s3_client.put_to_s3(s3_key, round_info)
            lock.release()
            logging.info("[%s] Released Zookeeper lock", self)
            self.zk.stop()
        else:
            with open(round_info_filename, "wb") as round_info_file:
                pickle.dump(round_info, round_info_file)
            self.lock.release()

        return data_shares[self.client_share_index]
