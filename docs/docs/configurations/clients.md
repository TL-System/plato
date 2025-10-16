!!! example "type"
    The type of the server.

    - `simple` a basic client who sends weight updates to the server.
    - `split_learning` a client following the Split Learning algorithm. When this client is used, `clients.do_test` in configuration should be set as `False` because in split learning, we conduct the test on the server.
    - `fedavg_personalized` a client saves its local layers before sending the shared global model to the server after local training.
    - `self_supervised_learning` a client to prepare the datasource for personalized learning based on self-supervised learning.
    - `mpc` a client that encrypts outbound model updates using multiparty computation processors.

!!! example "total_clients"
    The total number of clients in a training session.

!!! example "per_round"
    The number of clients selected in each round. It should be lower than `total_clients`.

!!! example "do_test"
    Whether or not the clients compute test accuracies locally using local testsets. Computing test accuracies locally may be useful in certain cases, such as personalized federated learning.

    Valid values: `true` or `false`

    !!! note "Note"
        If this setting is `true` and the configuration file has a `results` section, test accuracies of every selected client in each round will be logged in a `.csv` file.

!!! example "comm_simulation"
    Whether client-server communication should be simulated with reading and writing files. This is useful when the clients and the server are launched on the same machine and share a filesystem.

    Default value: `true`

    !!! example "compute_comm_time"
        When client-server communication is simulated, whether or not the transmission time — the time it takes for the payload to be completely transmitted to the server — should be computed with a pre-specified server bandwidth.

!!! example "speed_simulation"
    Whether or not the training speed of the clients are simulated. Simulating the training speed of the clients is useful when simulating *client heterogeneity*, where asynchronous federated learning may outperform synchronous federated learning.

    Valid values: `true` or `false`

    If `speed_simulation` is `true`, we need to specify the probability distribution used for generating a sleep time (in seconds per epoch) for each client, using the following settings:

    !!! example "random_seed"
        This random seed is used exclusively for generating the sleep time (in seconds per epoch).

        Default value: `1`

    !!! example "max_sleep_time"
        This is used to specify the longest possible sleep time in seconds.

        Default value: `60`

    !!! example "simulation_distribution"
        Parameters for simulating client heterogeneity in training speed. It has an embedded parameter `distribution`, which can be set to `normal` for the normal distribution, `zipf` for the Zipf distribution (which is discrete), or `pareto` for the Pareto distribution (which is continuous).

        For the normal distribution, we can specify `mean` for its mean value and `sd` for its standard deviation; for the Zipf distribution, we can specify `s`; and for the Pareto distribution, we can specify `alpha` to adjust how heavy-tailed it is. Here is an example:

        ```yaml
        speed_simulation: true
        simulation_distribution:
            distribution: pareto
            alpha: 1
        ```

!!! example "sleep_simulation"
    Should clients really go to sleep (`false`), or should we just simulate the sleep times (`true`)?

    Default value: `false`

    Simulating the sleep times — rather than letting clients go to sleep and measure the actual local training times including the sleep times — will be helpful to increase the speed of running the experiments, and to improve reproducibility, since every time the experiments run, the average training time will remain the same, and specified using the `avg_training_time` setting below.

    !!! example "avg_training_time"
        If we are simulating client training times, what is the average training time? When we are simulating the sleep times rather than letting clients go to sleep, we will not be able to use the measured wall-clock time for local training. As a result, we need to specify this value in lieu of the measured training time.

!!! example "outbound_processors"
    A list of processors for the client to apply on the payload before sending it out to the server. Multiple processors are permitted.

    - `outbound_feature_ndarrays` Convert PyTorch tensor features into NumPy arrays before sending to the server, for the benefit of saving a substantial amount of communication overhead if the feature dataset is large. Must be placed after `feature_unbatch`.
    - `model_deepcopy` Return a deepcopy of the state_dict to prevent changing internal parameters of the model within clients.
    - `model_randomized_response` Activate randomized response on model parameters, must also set `algorithm.epsilon` to activate.
    - `model_quantize` Quantize model parameters.
    - `model_quantize_qsgd` Quantize model parameters with QSGD.
    - `unstructured_pruning` Process unstructured pruning on model weights. The `model_compress` processor needs to be applied after it in the configuration file or the communication overhead will not be reduced.
    - `structured_pruning` Process structured pruning on model weights. The `model_compress` processor needs to be applied after it in the configuration file or the communication overhead will not be reduced.
    - `model_compress` Compress model parameters with `Zstandard` compression algorithm. Must be placed as the last processor if applied.
    - `model_encrypt` Encrypts the model parameters using homomorphic encryption.
    - `mpc_model_encrypt_additive` Encrypts model parameters using additive secret sharing for MPC.
    - `mpc_model_encrypt_shamir` Encrypts model parameters using Shamir secret sharing for MPC (adds misbehaving-client detection).

!!! example "mpc_debug_artifacts"
    When set to `true`, raw and encrypted client payloads are dumped to `mpc_data/` for inspection. Defaults to `false`.

!!! example "inbound_processors"
    A list of processors for the client to apply on the payload before receiving it from the server.

    - `model_decompress` Decompress model parameters. Must be placed as the first processor if `model_compress` is applied on the server side.
    - `model_decrypt` Decrypts the model parameters using homomorphic encryption.

!!! example "participating_clients_ratio"
    Percentage of clients participating in federated training out of all clients. The value should range from 0 to 1.
