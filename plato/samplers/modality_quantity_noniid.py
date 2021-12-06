"""
Samples data from a dataset, biased across modalities in an
 quantity-based nonIID fashion.

    Thus, the quantity-based modality non-IID can be achieved by just keeping
     one subset of modalities in each sample.

"""
import numpy as np

from plato.config import Config
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

        # Different clients should have a different bias across modalities
        np.random.seed(self.random_seed * int(client_id))

        # default, one sample holds only one modality
        per_client_modalties_size = Config(
        ).data.per_client_modalties_size if hasattr(
            Config().data, 'per_client_modalties_size') else 1

        assert per_client_modalties_size < len(modalities_name)

        # obtain the modalities that hold for this data
        self.subset_modalities = np.random.choice(
            modalities_name,
            per_client_modalties_size,
            replace=False,
        )

    def get(self):
        """Obtains the modality sampler.
            Note: the sampler here is utilized as the mask to
             remove modalities.
        """
        return self.subset_modalities

    def modality_size(self):
        """ Obtain the utilized modality size """
        return len(self.subset_modalities)
