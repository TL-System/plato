"""
Customize the inbound and outbound processors for MaskCrypt clients through callbacks.
"""

import maskcrypt_utils
import torch
import numpy as np
import os
from typing import Any, OrderedDict
from plato.processors import base
from plato.callbacks.client import ClientCallback
from plato.processors import model_encrypt, model_decrypt
from plato.config import Config

from plato.samplers import registry as samplers_registry


class ModelEstimateProcessor(base.Processor):
    """
    A client processor used to track the exposed model weights so far.
    """

    def __init__(self, client_id, is_init=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.is_init = is_init
        self.client_id = client_id

    def process(self, data: Any) -> Any:
        maskcrypt_utils.update_est(Config(), self.client_id, data, self.is_init)
        return data


class ModelTestProcessor(base.Processor):
    """
    A client processor used to track the exposed model weights so far.
    """

    def __init__(self, client_id, client, phase, **kwargs) -> None:
        super().__init__(**kwargs)
        self.client_id = client_id
        self.client = client
        self.phase = phase

        weight_shapes = {}
        para_nums = {}
        extract_model = client.algorithm.extract_weights()

        for key in extract_model.keys():
            weight_shapes[key] = extract_model[key].size()
            para_nums[key] = torch.numel(extract_model[key])

        self.weight_shapes = weight_shapes
        self.para_nums = para_nums

    def reconstruct(self, flatten_weights):
        vector_length = []
        for para_num in self.para_nums.values():
            vector_length.append(para_num)

        # Step 2: rebuild the original weight vector
        decrypted_weights = OrderedDict()
        plaintext_weights_vector = np.split(flatten_weights, np.cumsum(vector_length))[
            :-1
        ]
        weight_index = 0
        for name, shape in self.weight_shapes.items():
            decrypted_weights[name] = plaintext_weights_vector[weight_index].reshape(
                shape
            )
            try:
                decrypted_weights[name] = torch.from_numpy(decrypted_weights[name])
            except:
                # PyTorch does not exist, just return numpy array and handle it somewhere else.
                decrypted_weights[name] = decrypted_weights[name]
            weight_index = weight_index + 1
        return decrypted_weights

    def process(self, data: Any) -> Any:

        if self.phase == 1:
            model_weights = data
        else:
            new_est = maskcrypt_utils.update_est(Config(), self.client_id, data)
            model_weights = self.reconstruct(new_est)

        self.client.algorithm.load_weights(model_weights)
        testset = self.client.datasource.get_train_set()
        testset_sampler = samplers_registry.get(self.client.datasource, self.client_id)
        accuracy = self.client.trainer.test(testset, testset_sampler)
        print(f"============\n{self.client}, accuracy: {accuracy}\n==============\n")
        with open("results/%s_phase_%s.txt" % (os.getppid(), self.phase), "a") as f:
            f.write(
                "%s, \t %s, \t %s \n"
                % (self.client.current_round, self.client_id, accuracy)
            )
        return data


class MaskCryptCallback(ClientCallback):
    """
    A client callback that dynamically inserts encrypt and decrypt processors.
    """

    def on_inbound_received(self, client, inbound_processor):
        current_round = client.current_round
        if current_round % 2 != 0:
            # Update the exposed model weights from new global model
            inbound_processor.processors.append(
                ModelEstimateProcessor(
                    client_id=client.client_id, is_init=client.current_round == 1
                )
            )

            # Server sends model weights in odd rounds, add decrypt processor
            inbound_processor.processors.append(
                model_decrypt.Processor(
                    client_id=client.client_id,
                    trainer=client.trainer,
                    name="model_decrypt",
                )
            )

    def on_outbound_ready(self, client, report, outbound_processor):
        current_round = client.current_round
        if current_round % 2 == 0:
            outbound_processor.processors.append(
                ModelTestProcessor(client_id=client.client_id, client=client, phase=1)
            )
            # Clients send model weights to server in even rounds, add encrypt processor
            outbound_processor.processors.append(
                model_encrypt.Processor(
                    mask=client.final_mask,
                    client_id=client.client_id,
                    trainer=client.trainer,
                    name="model_encrypt",
                )
            )

            # Update the exposed model weights after encryption
            outbound_processor.processors.append(
                ModelEstimateProcessor(client_id=client.client_id)
            )

            outbound_processor.processors.append(
                ModelTestProcessor(client_id=client.client_id, client=client, phase=2)
            )
