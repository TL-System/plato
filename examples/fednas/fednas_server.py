import logging
import torch

from plato.config import Config
from plato.servers import fedavg

from Darts.model_search_local import MaskedNetwork
import torch.nn as nn
from fednas_tools import fuse_weight_gradient,extract_index,sample_mask

class Server(fedavg.Server):
    """Federated learning server using federated averaging."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(model, datasource, algorithm, trainer)

    def customize_server_response(self, server_response: dict) -> dict:
        """Customizes the server response with any additional information."""
        mask_normal = sample_mask(self.algorithm.model.model.alphas_normal)
        mask_reduce = sample_mask(self.algorithm.model.model.alphas_reduce)
        self.algorithm.mask_normal=mask_normal
        self.algorithm.mask_reduce=mask_reduce
        server_response['mask_normal']=mask_normal.numpy().tolist()
        server_response['mask_reduce']=mask_reduce.numpy().tolist()
        return server_response

    async def _process_reports(self):
        """Process the client reports by aggregating their weights."""
        weights_received = [update.payload for update in self.updates]
        mask_normals=[update.report.mask_normal for update in self.updates]
        mask_reduces = [update.report.mask_reduce for update in self.updates]
        self.callback_handler.call_event("on_weights_received", self, weights_received)

        # NAS aggregation
        client_models=[]
        epoch_index_normal = []
        epoch_index_reduce = []

        for i,payload in enumerate(weights_received):
            mask_normal=torch.tensor(mask_normals[i])
            mask_reduce=torch.tensor(mask_reduces[i])
            client_model = MaskedNetwork(Config().parameters.model.C, Config().parameters.model.num_classes,
                                         Config().parameters.model.layers, nn.CrossEntropyLoss(), mask_normal,
                                         mask_reduce)
            client_model.load_state_dict(weights_received[i],strict=True)
            client_models.append(client_model)
            index_normal = extract_index(mask_normal)
            index_reduce = extract_index(mask_reduce)
            epoch_index_normal.append(index_normal)
            epoch_index_reduce.append(index_reduce)
        fuse_weight_gradient(self.trainer.model.model,client_models)


        # Testing the global model accuracy
        if hasattr(Config().server, "do_test") and not Config().server.do_test:
            # Compute the average accuracy from client reports
            self.accuracy = self.accuracy_averaging(self.updates)
            logging.info(
                "[%s] Average client accuracy: %.2f%%.", self, 100 * self.accuracy
            )
        else:
            # Testing the updated model directly at the server

            self.accuracy = self.trainer.test(self.testset, self.testset_sampler)
        self.trainer.model.step([self.accuracy],epoch_index_normal, epoch_index_reduce)

        if hasattr(Config().trainer, "target_perplexity"):
            logging.info("[%s] Global model perplexity: %.2f\n", self, self.accuracy)
        else:
            logging.info(
                "[%s] Global model accuracy: %.2f%%\n", self, 100 * self.accuracy
            )

        await self.wrap_up_processing_reports()

    async def wrap_up(self):
        await super().wrap_up()
        logging.info("[%s] geneotypes: %s\n", self,self.trainer.model.model.genotype())