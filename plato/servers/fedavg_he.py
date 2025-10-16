"""Server for homomorphic-encrypted FedAvg aggregation."""

import torch

from plato.servers import fedavg
from plato.servers.strategies.aggregation import FedAvgHEAggregationStrategy
from plato.utils import homo_enc


class Server(fedavg.Server):
    """
    Federated learning server using federated averaging to aggregate updates after homomorphic
    encryption.
    """

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        aggregation_strategy=None,
        client_selection_strategy=None,
    ):
        if aggregation_strategy is None:
            aggregation_strategy = FedAvgHEAggregationStrategy()

        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
            aggregation_strategy=aggregation_strategy,
            client_selection_strategy=client_selection_strategy,
        )

        # Keep the composable server context intact; store CKKS context separately.
        self.he_context = homo_enc.get_ckks_context()
        self.encrypted_model = None
        self.weight_shapes = {}
        self.para_nums = {}

    def configure(self) -> None:
        """Configure the model information like weight shapes and parameter numbers."""
        super().configure()

        extract_model = self.trainer.model.cpu().state_dict()

        for key in extract_model.keys():
            self.weight_shapes[key] = extract_model[key].size()
            self.para_nums[key] = extract_model[key].numel()

        self.encrypted_model = homo_enc.encrypt_weights(
            extract_model, True, self.he_context, []
        )

    def customize_server_payload(self, payload):
        """Server can only send the encrypted aggreagtion result to clients."""
        return self.encrypted_model

    def _fedavg_hybrid(self, updates, weights_received):
        """Aggregate the model updates in the hybrid form of encrypted and unencrypted weights."""
        deserialized = [
            homo_enc.deserialize_weights(payload, self.he_context)
            for payload in weights_received
        ]
        unencrypted_weights = [
            homo_enc.extract_encrypted_model(x)[0] for x in deserialized
        ]
        encrypted_weights = [
            homo_enc.extract_encrypted_model(x)[1] for x in deserialized
        ]
        # Assert the encrypted weights from all clients are aligned
        indices = [homo_enc.extract_encrypted_model(x)[2] for x in deserialized]
        for i in range(1, len(indices)):
            assert indices[i] == indices[0]
        encrypt_indices = indices[0]

        # Extract the total number of samples
        self.total_samples = sum(update.report.num_samples for update in updates)

        # Perform weighted averaging on unencrypted and encrypted weights
        unencrypted_avg_update = self.trainer.zeros(unencrypted_weights[0].size)
        encrypted_avg_update = None

        for i, (unenc_w, enc_w) in enumerate(
            zip(unencrypted_weights, encrypted_weights)
        ):
            report = updates[i].report
            num_samples = report.num_samples

            unencrypted_avg_update += unenc_w * (num_samples / self.total_samples)
            if enc_w is not None:
                if encrypted_avg_update is None:
                    encrypted_avg_update = enc_w * 0
                encrypted_avg_update += enc_w * (num_samples / self.total_samples)

        if len(encrypt_indices) == 0:
            encrypted_avg_update = None

        # Ensure consistent serialization types
        unencrypted_vector = unencrypted_avg_update
        if hasattr(unencrypted_vector, "detach"):
            unencrypted_vector = unencrypted_vector.detach().cpu().numpy()

        return homo_enc.wrap_encrypted_model(
            unencrypted_vector, encrypted_avg_update, encrypt_indices
        )
