"""
A personalized federated learning trainer using FedRep.

"""

from statistics import mode
import warnings

warnings.simplefilter("ignore")

import torch
from tqdm import tqdm
import tracking
import pers_basic
from plato.config import Config
from optimizers import get_target_optimizer


class Trainer(pers_basic.Trainer):
    """A personalized federated learning trainer using the FedRep algorithm."""

    def freeze_model(self, model, param_prefix=None):
        for name, param in model.named_parameters():
            if param_prefix is not None and param_prefix in name:
                param.requires_grad = False

    def active_model(self, model, param_prefix=None):
        for name, param in model.named_parameters():
            if param_prefix is not None and param_prefix in name:
                param.requires_grad = True

    def perform_evaluation_op(self, to_eval_data_loader, defined_model):

        # Define the test phase of the eval stage
        acc_meter = tracking.AverageMeter(name="Accuracy")
        defined_model.eval()
        defined_model.to(self.device)
        correct = 0

        encoded_samples = list()
        loaded_labels = list()

        acc_meter.reset()
        for _, (examples, labels) in enumerate(to_eval_data_loader):
            examples, labels = examples.to(self.device), labels.to(self.device)
            with torch.no_grad():
                # preds = self.personalized_model(examples).argmax(dim=1)

                # features = defined_model.encoder(examples)
                preds = defined_model(examples).argmax(dim=1)

                correct = (preds == labels).sum()
                acc_meter.update(correct / preds.shape[0], labels.size(0))

                encoded_samples.append(preds)
                loaded_labels.append(labels)

        accuracy = acc_meter.average

        test_outputs = {
            "accuracy": accuracy,
            "encoded_samples": encoded_samples,
            "loaded_labels": loaded_labels,
        }

        return test_outputs

    def pers_train_one_epoch(
        self,
        config,
        pers_optimizer,
        lr_schedule,
        loss_criterion,
        train_loader,
        epoch_loss_meter,
    ):
        """Performing one epoch of learning for the personalization."""

        epoch_loss_meter.reset()
        self.personalized_model.train()
        self.personalized_model.to(self.device)

        pers_epochs = config["pers_epochs"]
        epoch = self.current_epoch

        local_progress = tqdm(
            train_loader, desc=f"Epoch {epoch}/{pers_epochs+1}", disable=True
        )

        self.freeze_model(self.personalized_model, param_prefix="encoder")
        self.active_model(self.personalized_model, param_prefix="clf_fc")

        for _, (examples, labels) in enumerate(local_progress):
            examples, labels = examples.to(self.device), labels.to(self.device)
            # Clear the previous gradient
            pers_optimizer.zero_grad()

            # Perfrom the training and compute the loss
            preds = self.personalized_model(examples)
            loss = loss_criterion(preds, labels)

            # Perfrom the optimization
            loss.backward()
            pers_optimizer.step()

            # Update the epoch loss container
            epoch_loss_meter.update(loss, labels.size(0))

            local_progress.set_postfix(
                {
                    "lr": lr_schedule,
                    "loss": epoch_loss_meter.val,
                    "loss_avg": epoch_loss_meter.average,
                }
            )

        return epoch_loss_meter

    def pers_train_run_start(
        self,
        config,
        **kwargs,
    ):
        """The customize behavior before performing one epoch of personalized training.
        By default, we need to save the encoded data, the accuracy, and the model when possible.
        """
        current_round = config["current_round"]
        eval_outputs, _ = super().pers_train_run_start(config, **kwargs)

        # self.checkpoint_encoded_samples(
        #     encoded_samples=eval_outputs["encoded_samples"],
        #     encoded_labels=eval_outputs["loaded_labels"],
        #     current_round=current_round,
        #     epoch=self.current_epoch,
        #     run_id=None,
        #     encoded_type="testEncoded",
        # )

        return eval_outputs, _

    def pers_train_epoch_end(
        self,
        config,
        **kwargs,
    ):
        current_round = config["current_round"]
        eval_outputs = super().pers_train_epoch_end(config, **kwargs)

        # if eval_outputs:
        #     self.checkpoint_encoded_samples(
        #         encoded_samples=eval_outputs["encoded_samples"],
        #         encoded_labels=eval_outputs["loaded_labels"],
        #         current_round=current_round,
        #         epoch=self.current_epoch,
        #         run_id=None,
        #         encoded_type="testEncoded",
        #     )

        return eval_outputs

    def get_optimizer(self, model, stage_prefix=None):
        model_name = Config().trainer.model_name
        if "t2t_vit" in model_name:
            _, optim_params = get_target_optimizer(stage_prefix)
            optimizer = torch.optim.SGD
            parameters = [
                {"params": model.model.tokens_to_token.parameters(), "lr": 5e-4},
                {"params": model.model.blocks.parameters(), "lr": 5e-4},
                {"params": model.model.head.parameters()},
            ]
            return optimizer(params=parameters, **optim_params)
        return super().get_optimizer(model, stage_prefix)
