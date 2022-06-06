"""
Samples data from a dataset, biased across modalities in an
 independent and identically distributed (IID) fashion.

    Thus, all modalities of one sample are utilized as the input.

    There is no difference between the train sampler and test sampler.
"""
import numpy as np

from plato.samplers import base


class Sampler(base.Sampler):
    """Create a data sampler for each client to use a randomly divided partition of the
        dataset."""
    def __init__(self, datasource, client_id):
        super().__init__()

        self.client_id = client_id
        if hasattr(datasource, "get_modality_name"):
            modalities_name = datasource.get_modality_name()
        else:  # default: it only contains image data
            modalities_name = ["rgb"]

        np.random.seed(self.random_seed)

        # obtain the modalities that hold for this data
        self.subset_modalities = modalities_name

    def get(self):
        """Obtains the modality sampler.
            Note: the sampler here is utilized as the mask to
             remove modalities.
        """
        return self.subset_modalities

    def modality_size(self):
        """ Obtain the utilized modality size """
        return len(self.subset_modalities)
