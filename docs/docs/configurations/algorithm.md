!!! example "type"
    Aggregation algorithm.

    The input should be:

    - `fedavg` the federated averaging algorithm
    - `split_learning` the Split Learning algorithm
    - `fedavg_personalized` the personalized federated learning algorithm
    - `pfedgraph` the Personalized Federated Learning with Inferred Collaboration Graphs algorithm

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

!!! example "pfedgraph"
    Configuration for pFedGraph.

    !!! example "pfedgraph_alpha"
        Hyper-parameter controlling the collaboration graph update.

        Default value: `0.8`

    !!! example "pfedgraph_lambda"
        Regularization strength for cosine similarity in the local objective.

        Default value: `0.01`

    !!! example "pfedgraph_similarity_metric"
        Similarity metric scope for graph inference. Use `all` for all parameters
        or `fc` to focus on the final fully-connected layers.

        Default value: `all`

    !!! example "pfedgraph_similarity_layers"
        Optional list of layer name substrings to use when computing model
        similarity for graph inference. Overrides `pfedgraph_similarity_metric`
        when provided.
