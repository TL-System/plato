"""
Assign modalities to clients in an independent and identically distributed fashion.
We achieve this by directly working on each sample.
Thus, the non-IID can be achieved just keep each modality in each sample.
"""
import numpy as np

from plato.samplers.multimodal import modality_base


class Sampler(modality_base.Sampler):
    """Create a data sampler for each client to use a randomly divided partition of the
        dataset."""
    def __init__(self, datasource, client_id):
        super().__init__()
        self.client_id = client_id
        modalities_name = datasource.get_modality_name()

        np.random.seed(self.random_seed)

        # obtain the modalities that hold for this data
        self.subset_modalities = modalities_name

    def get(self):
        """Obtains the modality sampler. """
        return self.subset_modalities

    def trainset_size(self):
        """Returns the length of the dataset after sampling. """
        return len(self.subset_indices)

    def modality_size(self):
        return len(self.subset_modalities)
