"""
The training and testing loop.
"""

import logging
import torch
from training import optimizers

# CUDA settings
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def extract_weights(model):
    """Extract weights from a model passed in as a parameter."""
    weights = []
    for name, weight in model.to(torch.device('cpu')).named_parameters():
        if weight.requires_grad:
            weights.append((name, weight.data))

    return weights


def load_weights(model, weights):
    """Load the model weights passed in as a parameter."""
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight

    model.load_state_dict(updated_state_dict, strict=False)


def train(model, trainset, config):
    """The main training loop for each client in a federated learning workload.

    Arguments:
      model: The model to train. Must be a models.base.Model subclass.
      trainset: The training dataset.
      config: Training hyperparameters.
    """
    model.to(device)
    model.train()

    log_interval = 10
    batch_size = config.training.batch_size
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    epochs = config.training.epochs
    optimizer = optimizers.get_optimizer(config, model)

    for epoch in range(1, epochs + 1):
        for batch_id, (examples, labels) in enumerate(train_loader):
            examples, labels = examples.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = model.loss_criterion(model(examples), labels)
            loss.backward()
            optimizer.step()

            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, epochs, loss.item()))


def test(model, testset, batch_size):
    """Testing the model using the provided test dataset."""
    model.to(device)
    model.eval()
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            examples, labels = data
            examples, labels = examples.to(device), labels.to(device)
            outputs = model(examples)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

    return accuracy
