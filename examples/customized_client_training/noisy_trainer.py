import torch
import logging
from plato.config import Config
from plato.trainers import basic
import random
def get_coreset_selection():
    """Get coreset selection based on the configuration file."""
    coreset_selection = (
        Config().clients.coreset_selection
        if hasattr(Config().clients, "coreset_selection")
        else None
    )

    if coreset_selection is None:
        logging.info(f"No coreset selection is applied.")
        return lambda x: x

    if coreset_selection in registered_selections:
        registered_attack = registered_selections[coreset_selection]
        logging.info(f"Clients perform {coreset_selection}.")
        return registered_attack

    raise ValueError(f"No such selection: {coreset_selection}")

class Trainer(basic.Trainer):
    """The federated learning trainer for the noisy client."""
    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """
        Creates an instance of the trainloader.

        Arguments:
        batch_size: the batch size.
        trainset: the training dataset.
        sampler: the sampler for the trainloader to use.
        """
        
        local_data_loader =  torch.utils.data.DataLoader(
            dataset=trainset, shuffle=False, batch_size=batch_size, sampler=sampler
        )
        logging.info(f"the len of local data loader is : %s", len(local_data_loader))
        
        # get label correction method 
        # label_correction = get_label_correction()
        # correct local label
        # train_dataloader_corrected = label_correction(local_data_loader)

        # get coreset selection method
        coreset_selection = get_coreset_selection()

        # get coreset via selection method
        train_dataloader_coreset = coreset_selection(local_data_loader)

        return train_dataloader_coreset

def random_coreset(local_data_loader):
    dropout_probability = 0.1  # 10% chance of dropping a batch
    train_dataloader_coreset = []
    for batch_id, (examples, labels) in enumerate(local_data_loader):
        # Randomly decide whether to skip this batch
        if random.random() < dropout_probability:
            continue  # Skip this iteration and move to the next batch
        else:
            train_dataloader_coreset.append((examples,labels))
        logging.info(f"len of coreset: %s",len(train_dataloader_coreset))
    return train_dataloader_coreset


registered_selections = {
    "random": random_coreset,
}
