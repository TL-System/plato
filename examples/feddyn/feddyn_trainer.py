import torch

from plato.config import Config
from plato.trainers import basic


class Trainer(basic.Trainer):

    def perform_forward_and_backward_passes(self, config, examples, labels):
        """Perform forward and backward passes in the training loop.

        Arguments:
        config: the configuration.
        examples: data samples in the current batch.
        labels: labels in the current batch.

        Returns: loss values after the current batch has been processed.
        """
        # self.optimizer.zero_grad()

        # outputs = self.model(examples)

        # loss = self._loss_criterion(outputs, labels)
        # self._loss_tracker.update(loss, labels.size(0))

        # if "create_graph" in config:
        #     loss.backward(create_graph=config["create_graph"])
        # else:
        #     loss.backward()

        # self.optimizer.step()

        # return loss

        alpha_coef = config["parameters"]["alpha_coef"]
        avg_mdl_param = config["parameters"]["avg_mdl_param"]
        local_grad_vector = config["parameters"]["local_grad_vector"]

        model = self.model.to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        
        epoch_loss = 0

        self.optimizer.zero_grad()

        batch_x = examples.to(self.device)
        batch_y = labels.to(self.device)
        
        y_pred = model(batch_x)
        
        ## Get f_i estimate 
        loss_f_i = loss_fn(y_pred, batch_y.reshape(-1).long())
        loss_f_i = loss_f_i / list(batch_y.size())[0]
        
        # Get linear penalty on the current parameter estimates
        local_par_list = None
        for param in model.parameters():
            if not isinstance(local_par_list, torch.Tensor):
            # Initially nothing to concatenate
                local_par_list = param.reshape(-1)
            else:
                local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
        
        loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + local_grad_vector))
        loss = loss_f_i + loss_algo

        self.optimizer.step()
        
        return loss
