import os

os.environ[
    'config_file'] = 'examples/onestep_ml_pfl/os_ml_pfl_CIFAR10_lenet5.yml'

from utils import verify_working_correcness, verify_client_local_data_correcness, \
    verify_difference_between_clients

from plato.config import Config
from plato.datasources.cifar10 import DataSource
from ml_pfl_sampler import Sampler

if __name__ == "__main__":
    _ = Config()

    print(Config().data.per_client_classes_size)

    cifar10_datasource = DataSource()

    client_id = 1
    verify_working_correcness(Sampler,
                              dataset_source=cifar10_datasource,
                              client_id=client_id,
                              num_of_batches=3,
                              batch_size=5)
    print("-" * 20)
    verify_flag = verify_client_local_data_correcness(
        Sampler,
        dataset_source=cifar10_datasource,
        client_id=client_id,
        num_of_iterations=2,
        batch_size=5,
        is_presented=False)
    if verify_flag:
        print(
            ("Ensure that the local data assigned to the client {} maintains \
                the same local data in different runs").format(client_id))
    print("-" * 20)

    selected_clients, clients_classes_info, \
        clients_classes_sample_info = verify_difference_between_clients(
        list(range(50)),
        Sampler,
        cifar10_datasource,
        num_of_batches=None,
        batch_size=5,
        is_presented=False)

    print("Selected clients: ", selected_clients)
    for client_id in list(clients_classes_info.keys()):
        print(("Client {}, classes {}, samples {}").format(
            client_id, clients_classes_info[client_id],
            clients_classes_sample_info[client_id]))
