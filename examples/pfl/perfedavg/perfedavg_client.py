"""
The implementation of Per-FedAvg method based on the plato's
pFL code.

Alireza Fallah, et.al, Personalized federated learning with theoretical guarantees:
A model-agnostic meta-learning approach, NIPS2020.
 https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html

Official code: None
Third-part code: https://github.com/jhoon-oh/FedBABU


"""

from ..bases import simple_personalized


class Client(simple_personalized.Client):
    """A Per-FedAvg federated learning client."""

    def _load_payload(self, server_payload) -> None:
        """Load the server model onto this client.
        Each client will directly receive the global model
        as the local model to perform the local update
        """

        # assign the received payload to the local model
        self.algorithm.load_weights(server_payload)
