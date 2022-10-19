"""
Pisces: an asynchronous client selection and server aggregation algorithm.

Reference:

Z. Jiang, B. Wang, B. Li, B. Li. "Pisces: Efficient Federated Learning via Guided Asynchronous
Training," in Proceedings of ACM Symposium on Cloud Computing (SoCC), 2022.

URL: https://arxiv.org/abs/2206.09264
"""

from plato.trainers import basic, loss_criterion


class Trainer(basic.Trainer):
    """The federated learning trainer for the Pisces client."""

    def process_loss(self, outputs, labels):
        """Returns the loss and records per_batch loss values."""
        loss_func = loss_criterion.get()
        per_batch_loss = loss_func(outputs, labels)

        # Stores the per_batch loss value
        self.run_history.update_metric(
            "train_batch_loss", per_batch_loss.cpu().detach().numpy()
        )

        return per_batch_loss

    def get_loss_criterion(self):
        """Returns the loss criterion."""
        return self.process_loss
