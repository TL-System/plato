"""
A federated learning server using federated averaging to aggregate updates after homomorphic encryption.
"""
from functools import reduce
from plato.servers import fedavg
from plato.utils import homo_enc


class Server(fedavg.Server):
    """
    Federated learning server using federated averaging to aggregate updates after homomorphic
    encryption.
    """

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

        self.context = homo_enc.get_ckks_context()
        self.encrypted_model = None
        self.weight_shapes = {}
        self.para_nums = {}

    def configure(self) -> None:
        """Configure the model information like weight shapes and parameter numbers."""
        super().configure()

        extract_model = self.trainer.model.cpu().state_dict()

        for key in extract_model.keys():
            self.weight_shapes[key] = extract_model[key].size()
            self.para_nums[key] = reduce(lambda a, b: a * b, self.weight_shapes[key])

        self.encrypted_model = homo_enc.encrypt_weights(
            extract_model, True, self.context, []
        )

    def customize_server_payload(self, payload):
        """Server can only send the encrypted aggreagtion result to clients."""
        return self.encrypted_model

    # pylint: disable=unused-argument
    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """Aggregate the model updates and decrypt the result for evaluation purpose."""
        self.encrypted_model = self._fedavg_hybrid(updates)

        # Decrypt model weights for test accuracy
        decrypted_weights = homo_enc.decrypt_weights(
            self.encrypted_model, self.weight_shapes, self.para_nums
        )
        # Serialize the encrypted weights after decryption
        self.encrypted_model["encrypted_weights"] = self.encrypted_model[
            "encrypted_weights"
        ].serialize()

        return decrypted_weights

    def _fedavg_hybrid(self, updates):
        """Aggregate the model updates in the hybrid form of encrypted and unencrypted weights."""
        weights_received = [
            homo_enc.deserialize_weights(update.payload, self.context)
            for update in updates
        ]
        unencrypted_weights = [
            homo_enc.extract_encrypted_model(x)[0] for x in weights_received
        ]
        encrypted_weights = [
            homo_enc.extract_encrypted_model(x)[1] for x in weights_received
        ]
        # Assert the encrypted weights from all clients are aligned
        indices = [homo_enc.extract_encrypted_model(x)[2] for x in weights_received]
        for i in range(1, len(indices)):
            assert indices[i] == indices[0]
        encrypt_indices = indices[0]

        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging on unencrypted weights
        unencrypted_avg_update = self.trainer.zeros(unencrypted_weights[0].size)
        encrypted_avg_update = self.trainer.zeros(encrypted_weights[0].size())

        for i, (unenc_w, enc_w) in enumerate(
            zip(unencrypted_weights, encrypted_weights)
        ):
            report = updates[i].report
            num_samples = report.num_samples

            unencrypted_avg_update += unenc_w * (num_samples / self.total_samples)
            encrypted_avg_update += enc_w * (num_samples / self.total_samples)

        if len(encrypt_indices) == 0:
            # No weights are encrypted, set to None
            encrypted_avg_update = None

        return homo_enc.wrap_encrypted_model(
            unencrypted_avg_update, encrypted_avg_update, encrypt_indices
        )
