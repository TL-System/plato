"""
Having a registry of all available wrappers is convenient for retrieving an instance
from the defined dataset for specific usages, such as constrastive learning.
"""

import logging
from collections import OrderedDict

from plato.config import Config

from plato.datasources.contrastive_data_wrapper import (ContrastiveDataWrapper)

registered_datasources_wrapper = OrderedDict([('ContrastiveWrapper',
                                               ContrastiveDataWrapper)])


def get(datasource, augment_transformer=None):
    """Get the data source with the provided name."""
    datasource_wrapper_name = Config().data.data_wrapper

    logging.info("Data source wrapper: %s", datasource_wrapper_name)

    if datasource_wrapper_name in list(registered_datasources_wrapper.keys()):
        return registered_datasources_wrapper[datasource_wrapper_name](
            datasource, augment_transformer)
    else:
        raise ValueError(
            f'No such data source wrapper: {datasource_wrapper_name}')
