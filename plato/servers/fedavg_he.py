"""
A federated learning server using federated averaging to aggregate updates after homomorphic encryption.
"""
from functools import reduce
from plato.servers import fedavg
from plato.utils import homo_enc


class Server(fedavg.Server):
    """Federated learning server using federated averaging to aggregate updates after homomorphic encryption."""

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

    def configure(self):
        """Configure"""
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

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        # Extract the
        weights_received = [
            homo_enc.deserialize_weights(update.payload, self.context)
            for update in updates
        ]
        # Check if all the weights are encrypted
        encrypt_indices = [weights["encrypt_indices"] for weights in weights_received]
        for indices in encrypt_indices:
            assert indices is None

        # Aggregate the encrypted weights
        encrypted_weights = [
            weights["encrypted_weights"] for weights in weights_received
        ]

        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging
        encrypted_avg_update = self.trainer.zeros(encrypted_weights[0].size())

        for i, weights in enumerate(encrypted_weights):
            report = updates[i].report
            num_samples = report.num_samples
            encrypted_avg_update += weights * (num_samples / self.total_samples)

        # Wrap up the aggregation result
        self.encrypted_model = {}
        self.encrypted_model["encrypted_weights"] = encrypted_avg_update
        self.encrypted_model["encrypt_indices"] = None
        self.encrypted_model["unencrypted_weights"] = None

        # Decrypt the model weights to evaluate accuracy
        decrypted_weights = homo_enc.decrypt_weights(
            self.encrypted_model, self.weight_shapes, self.para_nums
        )
        self.encrypted_model["encrypted_weights"] = encrypted_avg_update.serialize()
        return decrypted_weights
