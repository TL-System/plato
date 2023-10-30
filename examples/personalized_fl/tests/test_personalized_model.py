"""
Test whether personalized models of clients are initialized correctly.
1. the number defined personalized models only equals to the processes of Plato
2. each client has its unique personalized model due to the reinitialization 
of the model based on the `client id` as the random seed.
"""

from collections import defaultdict

from pflbases import personalized_client
from pflbases import personalized_trainer
from pflbases import trainer_utils


class AppendDict(defaultdict):
    """A dict allowing to append values to keys."""

    def __init__(self):
        super().__init__(list)


def flatten_nested_list(nested_list):
    """Flatten a nested list."""
    return [item for sublist in nested_list for item in sublist]


class PersonalizedModelsTest:
    """A unit test whether the clients with different ids can create personalized models
    correctly."""

    def __init__(self):
        self.client = personalized_client.Client(trainer=personalized_trainer.Trainer)
        self.test_ids = [1, 1, 1, 3, 5, 6, 10, 50, 100, 500]

    def test_personalized_models(self):
        """Test whether each client has the unique personalized model.
        0. The actual personalized model will be defined once, i.e., be assigned
            to one memory.
        1. Clients with same ids have the same personalized models, i.e.,
        when ids are the same only one personalized model will be initialized.
        2. Clients with different ids have different personalized models.
        """

        models_mem_address = AppendDict()
        models_unique_id = AppendDict()
        models_statistic = AppendDict()

        for id_n in self.test_ids:
            self.client.client_id = id_n
            self.client.configure()

            statistic = trainer_utils.compute_model_statistics(
                self.client.trainer.personalized_model
            )
            address = hex(id(self.client.trainer.personalized_model))
            model_id = id(self.client.trainer.personalized_model)

            models_mem_address[id_n].append(address)
            models_unique_id[id_n].append(model_id)
            models_statistic[id_n].append(statistic)

        # ensure all personalized models correspond to the same memory address
        # as the model is defined only once
        assert len(set(flatten_nested_list(models_mem_address.values()))) == 1
        assert len(set(flatten_nested_list(models_unique_id.values()))) == 1

        # Ensure point 1
        models_sta_values = AppendDict()
        for client_id in models_statistic:
            client_statistics = models_statistic[client_id]
            base_statistic = client_statistics[0]
            assert all(sta == base_statistic for sta in client_statistics)
            for item in base_statistic:
                models_sta_values[item].append(base_statistic[item])

        # Ensure point 2
        for clients_sta in models_sta_values.values():
            assert len(set(clients_sta)) == len(clients_sta)


if __name__ == "__main__":
    unit_test = PersonalizedModelsTest()
    unit_test.test_personalized_models()
