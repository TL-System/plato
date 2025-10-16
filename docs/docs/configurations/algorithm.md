!!! example "type"
    Aggregation algorithm.

    The input should be:

    - `fedavg` the federated averaging algorithm
    - `split_learning` the Split Learning algorithm
    - `fedavg_personalized` the personalized federated learning algorithm

!!! example "cross_silo"
    Whether or not cross-silo training should be used.

    !!! example "total_silos"
        The total number of silos (edge servers). The input could be any positive integer.

    !!! example "local_rounds"
        The number of local aggregation rounds on edge servers before sending aggregated weights to the central server. The input could be any positive integer.

!!! example "fedavg_personalized"
    Whether or not the personalized training should be used.

    !!! example "local_layer_names"
        Local layers in a model should remain local at the clients during personalized FL training, and should not be aggregated at the server.

    !!! example "participating_clients_ratio"
        A float to show the proportion of clients participating in the federated training process. It is under `personalization`, which is a sub-config path that contains other personalized training parameters.

        Default value: `1.0`
