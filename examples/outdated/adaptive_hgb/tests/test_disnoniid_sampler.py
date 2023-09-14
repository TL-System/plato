"""
Test the distribution-based lable nonIID

"""

import os

os.environ['config_file'] = 'examples/adaptive_hgb/tests/sampler_config.yml'

from utils import verify_working_correctness, verify_client_local_data_correctness, \
    verify_difference_between_clients

from plato.config import Config
from plato.datasources.cifar10 import DataSource
from plato.samplers.multimodal.distribution_noniid import Sampler

if __name__ == "__main__":
    _ = Config()

    print(Config().data.per_client_classes_size)

    cifar10_datasource = DataSource()

    client_id = 1
    verify_working_correctness(Sampler,
                               dataset_source=cifar10_datasource,
                               client_id=client_id,
                               num_of_batches=3,
                               batch_size=5)
    print("-" * 20)
    verify_flag = verify_client_local_data_correctness(
        Sampler,
        dataset_source=cifar10_datasource,
        client_id=client_id,
        num_of_iterations=2,
        batch_size=5,
        is_presented=True)
    if verify_flag:
        print(("Ensure that the local data assigned to the client {} maintains\
                 the same in different runs").format(client_id))
    print("-" * 20)
    verify_difference_between_clients([0, 1, 2],
                                      Sampler,
                                      cifar10_datasource,
                                      num_of_batches=None,
                                      batch_size=5,
                                      is_presented=True)
