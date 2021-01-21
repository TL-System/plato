"""
This is the fednova trainer, used by both the client and the
server. Notably, the local iteration is appointed by the client.
"""
import torch
import logging
from config import Config
from trainers import mistnet, optimizers


class Trainer(mistnet.Trainer):
    """A trainer for fednova"""
    def train(self, trainset, cut_layer=None, iteration=1):

        trainset = mistnet.FeatureDataset(trainset)
        log_interval = 10
        batch_size = Config().trainer.batch_size
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        #iterations_per_epoch = np.ceil(len(trainset) / batch_size).astype(int)
        epochs = iteration  #Config().trainer.epochs

        # Initializing the optimizer
        optimizer = optimizers.get_optimizer(self.model)

        for epoch in range(1, epochs + 1):
            for batch_id, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(self.device), labels.to(
                    self.device)
                optimizer.zero_grad()
                if cut_layer is None:
                    loss = self.model.loss_criterion(self.model(examples),
                                                     labels)
                else:
                    outputs = self.model.forward_from(examples, cut_layer)
                    loss = self.model.loss_criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if batch_id % log_interval == 0:
                    logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                        epoch, epochs, loss.item()))
