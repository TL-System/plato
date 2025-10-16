!!! example "type"
    The type of the server.

    - `fedavg` a Federated Averaging (FedAvg) server.
    - `fedavg_cross_silo` a Federated Averaging server that handles cross-silo federated learning by interacting with edge servers rather than with clients directly. When this server is used, `algorithm.type` must be `fedavg`.
    - `fedavg_gan` a Federated Averaging server that handles Generative Adversarial Networks (GANs).
    - `fedavg_he` a Federated Averaging server that handles model updates after homomorphic encryption. When this server is used, the clients need to enable inbound processor `model_decrypt` to decrypt the global model from server, and outbound processor `model_encrypt` to encrypt the model updates.
    - `fedavg_personalized` a Federated Averaging server that supports all-purpose personalized federated learning by controlling when and which group of clients are to perform local personalization.
    - `fedavg_mpc_additive` a Federated Averaging server that reconstructs additive MPC shares before aggregation. Requires clients of type `mpc` with the `mpc_model_encrypt_additive` processor.
    - `fedavg_mpc_shamir` a Federated Averaging server that reconstructs Shamir MPC shares before aggregation. Requires clients of type `mpc` with the `mpc_model_encrypt_shamir` processor.
    - `split_learning` a Split Learning server that supports training different kinds of models in split learning framework. When this server is used, the `clients.per_round` in the configuration should be set to 1. Users should define the rules for updating models weights before cut from the clients to the server in the callback function `on_update_weights_before_cut`, depending on the specific model they use.
    - `fedavg_personalized` a personalized federated learning server that starts from a number of regular rounds of federated learning. In these regular rounds, only a subset of the total clients can be selected to perform the local update (the ratio of which is a configuration setting). After all regular rounds are completed, it starts a final round of personalization, where a selected subset of clients perform local training using their local dataset.

!!! example "address"
    The address of the central server, such as `127.0.0.1`.

!!! example "port"
    The port number of the central server, such as `8000`.

!!! example "disable_clients"
    If this optional setting is `true`, the server will not launched client processes on the same physical machine. This is useful when the server is deployed in the cloud and connected to by remote clients.

!!! example "s3_endpoint_url"
    The endpoint URL for an S3-compatible storage service, used for transferring payloads between clients and servers.

!!! example "s3_bucket"
    The bucket name for an S3-compatible storage service, used for transferring payloads between clients and servers.

!!! example "zk_address"
    The address of the ZooKeeper service used to coordinate MPC clients when S3 storage is enabled.

!!! example "zk_port"
    The port of the ZooKeeper service used to coordinate MPC clients when S3 storage is enabled.

!!! example "random_seed"
    The random seed used for selecting clients (and sampling the test dataset on the server, if needed) so that experiments are reproducible.

!!! example "ping_interval"
    The time interval in seconds at which the server pings the client.

    Default value: `3600`

!!! example "ping_timeout"
    The time in seconds that the client waits for the server to respond before disconnecting.

    Default value: `3600`

!!! example "synchronous"
    Whether training session should operate in synchronous (`true`) or asynchronous (`false`) mode.

!!! example "periodic_interval"
    The time interval for a server operating in asynchronous mode to aggregate received updates. Any positive integer could be used for `periodic_interval`.

    Default value: `5` seconds

    !!! note "Note"
        This is only used when we are not simulating the wall-clock time using the `simulate_wall_time` setting below.

!!! example "simulate_wall_time"
    Whether or not the wall clock time on the server is simulated. This is useful when clients train in batches, rather than concurrently, due to limited resources (such as a limited amount of CUDA memory on the GPUs).

!!! example "staleness_bound"
    In asynchronous mode, whether or not we should wait for clients who are behind the current round (*stale*) by more than this value. Any positive integer could be used for `staleness_bound`.

    Default value: `0`

!!! example "minimum_clients_aggregated"
    When operating in asynchronous mode, the minimum number of clients that need to arrive before aggregation and processing by the server. Any positive integer could be used for `minimum_clients_aggregated`.

    Default value: `1`

!!! example "minimum_edges_aggregated"
    When operating in asynchronous cross-silo federated learning, the minimum number of edge servers that need to arrive before aggregation and processing by the central server. Any positive integer could be used for `minimum_edges_aggregated`.

    Default value: `algorithm.total_silos`

!!! example "do_test"
    Whether the server tests the global model and computes the global accuracy or perplexity.

    Default value: `true`

!!! example "model_path"
    The path to the pretrained and trained models.

    Default value: `<base_path>/models/pretrained`, where `<base_path>` is specified in the `general` section.

!!! example "checkpoint_path"
    The path to temporary checkpoints used for resuming the training session.

    Default value: `<base_path>/checkpoints`, where `<base_path>` is specified in the `general` section.

!!! example "mpc_data_path"
    Directory where MPC coordination artefacts (e.g., `round_info`) and optional debug payloads are stored when running locally. Defaults to `<base_path>/mpc_data`.

!!! example "outbound_processors"
    A list of processors to apply on the payload before sending it out to the clients. Multiple processors are permitted.

    - `unstructured_pruning` Process unstructured pruning on model weights. The `model_compress` processor needs to be applied after it in the configuration file or the communication overhead will not be reduced.
    - `structured_pruning` Process structured pruning on model weights. The `model_compress` processor needs to be applied after it in the configuration file or the communication overhead will not be reduced.
    - `model_compress` Compress model parameters with `Zstandard` compression algorithm. Must be placed as the last processor if applied.

!!! example "inbound_processors"
    A list of processors to apply on the payload right after receiving. Multiple processors are permitted.

    - `model_decompress` Decompress model parameters. Must be placed as the first processor if `model_compress` is applied on the client side.
 `outbound_feature_ndarrays`.
    - `model_dequantize` Dequantize model parameters back to the 32-bit floating number format.
    - `model_dequantize_qsgd` Dequantize model parameters quantized with QSGD.

!!! example "mpc_shamir_threshold"
    Overrides the number of shares required to reconstruct Shamir-secret-shared tensors. Defaults to `number_of_selected_clients - 2`.

!!! example "downlink_bandwidth"
    The server's estimated downlink capacity (server to clients or central server to edge servers in cross-silo training) in Mbps, used for computing the transmission time (see `compute_comm_time` in the `clients` section).

    Default value: `100`

!!! example "uplink_bandwidth"
    The server's estimated uplink capacity (server to clients or central server to edge servers in cross-silo training) in Mbps, used for computing the transmission time (see `compute_comm_time` in the `clients` section).

    Default value: `100`

!!! example "edge_downlink_bandwidth"
    The edge server's estimated downlink capacity (an edge server to its clients) in Mbps, used for computing the transmission time (see `compute_comm_time` in the `clients` section).

    Default value: same as `downlink_bandwidth`

!!! example "edge_uplink_bandwidth"
    The edge server's estimated uplink capacity (an edge server to its clients) in Mbps, used for computing the transmission time (see `compute_comm_time` in the `clients` section).

    Default value: same as `uplink_bandwidth`

!!! example "do_personalization_interval"
    The round interval for a server commanding when to perform personalization.

    Default value: `0`, meaning that no personalization will be performed.

!!! example "do_personalization_group"
    The group of clients that is required by the server to perform personalization. There are three options, including "total", "participant", and "nonparticipant".

    Default value: `participant`, meaning the clients participating in training will be used to perform personalization.
