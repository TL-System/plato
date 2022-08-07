# Configuration Settings

In Plato, all configuration settings are read from a configuration file when the clients and the servers launch, and the configuration file follows the YAML format for the sake of simplicity and readability. This document introduces all the possible settings in the configuration file.

```{note}
Attributes in **bold** must be included in a configuration file, while attributes in *italic* only need to be included under certain conditions.
```

## general

```{admonition} base_path
The path prefix for datasets, models, checkpoints, and results.

The default value is `./`.
```

## clients

```{admonition} **type**
The type of the federated learning client. Valid values include `simple`, which represents a basic client who sends weight updates to the server; and `mistnet`, which is client following the MistNet algorithm.
```

```{admonition} **total_clients**
The total number of clients in a training session.
```

```{admonition} **per_round**
The number of clients selected in each round. It should be lower than `total_clients`.
```

````{admonition} do_test
Whether or not the clients compute test accuracies locally using local testsets. Computing test accuracies locally may be useful in certain cases, such as personalized federated learning. Valid values are `true` or `false`.

```{note}
If this setting is `true` and the configuration file has a `results` section, test accuracies of every selected client in each round will be logged in a `.csv` file.
```
````

````{admonition} comm_simulation
Whether client-server communication should be simulated with reading and writing files. This is useful when the clients and the server are launched on the same machine and share a filesystem. 

The default value is `true`.

```{admonition} compute_comm_time
When client-server communication is simulated, whether or not the transmission time — the time it takes for the payload to be completely transmitted to the server — should be computed with a pre-specified server bandwidth.
```

````

`````{admonition} speed_simulation
Whether or not the training speed of the clients are simulated. Simulating the training speed of the clients is useful when simulating *client heterogeneity*, where asynchronous federated learning may outperform synchronous federated learning. Valid values are `true` or `false`.

If `speed_simulation` is `true`, we need to specify the probability distribution used for generating a sleep time (in seconds per epoch) for each client, using the following setting:

```{admonition} random_seed
This random seed is used exclusively for generating the sleep time (in seconds per epoch). 

The default value is `1`.
```

```{admonition} max_sleep_time
This is used to specify the longest possible sleep time in seconds. 

The default value is `60`.
```

````{admonition} simulation_distribution
Parameters for simulating client heterogeneity in training speed. It has an embedded parameter `distribution`, which can be set to `normal` for the normal distribution, `zipf` for the Zipf distribution (which is discrete), or `pareto` for the Pareto distribution (which is continuous).

For the normal distribution, we can specify `mean` for its mean value and `sd` for its standard deviation; for the Zipf distribution, we can specify `s`; and for the Pareto distribution, we can specify `alpha` to adjust how heavy-tailed it is. Here is an example:

```yaml
speed_simulation: true
simulation_distribution: pareto
    distribution: pareto
    alpha: 1
```
````
`````

```{admonition} outbound_processors
A list of processors for the client to apply on the payload before sending it out to the server. Multiple processors are permitted.

- `feature_randomized_response` Activate randomized response on features for PyTorch MistNet, must also set `algorithm.epsilon` to activate. Must be placed before `feature_unbatch`.

- `feature_laplace` Add random noise with laplace distribution to features for PyTorch MistNet. Must be placed before `feature_unbatch`.

- `feature_gaussian` Add random noise with gaussian distribution to features for PyTorch MistNet. Must be placed before `feature_unbatch`.

- `feature_quantize` Quantize features for PyTorch MistNet. Must not be used together with `outbound_feature_ndarrays`.

- `feature_unbatch` Unbatch features for PyTorch MistNet clients, must use this processor for every PyTorch MistNet client before sending.

- `outbound_feature_ndarrays` Convert PyTorch tensor features into NumPy arrays before sending to the server, for the benefit of saving a substantial amount of communication overhead if the feature dataset is large. Must be placed after `feature_unbatch`.

- `model_deepcopy` Return a deepcopy of the state_dict to prevent changing internal parameters of the model within clients.

- `model_randomized_response` Activate randomized response on model parameters for PyTorch, must also set `algorithm.epsilon` to activate.

- `model_quantize` Quantize features for model parameters for PyTorch.

- `unstructured_pruning` Process unstructured pruning on model weights for PyTorch. The `model_compress` processor needs to be applied after it in the configuration file or the communication overhead will not be reduced.

- `structured_pruning` Process structured pruning on model weights for PyTorch. The `model_compress` processor needs to be applied after it in the configuration file or the communication overhead will not be reduced.

- `model_compress` Compress model parameters with `Zstandard` compression algorithm. Must be placed as the last processor if applied.
```

```{admonition} inbound_processors
A list of processors for the client to apply on the payload before receiving it from the server. Multiple processors are permitted.

- `model_decompress` Decompress model parameters. Must be placed as the first processor if `model_compress` is applied on the server side.
```

## server


```{admonition} type
The type of the server.

- `fedavg` a Federated Averaging (FedAvg) server.
- `fedavg_cross_silo` a Federated Averaging server that handles cross-silo federated learning by interacting with edge servers rather than with clients directly. When this server is used, `algorithm.type` must be `fedavg`.
- `mistnet` a MistNet server.
- `fedavg_gan` a Federated Averaging server that handles Generative Adversarial Networks (GANs).
```

```{admonition} address
The address of the central server.

e.g., `127.0.0.1`
```

```{admonition} port
The port number of the central server.

e.g., `8000`, `8005`
```

```{admonition} disable_clients
If this optional setting is enabled as `true`, the server will not launched client processes on the same machine.
```

```{admonition} s3_endpoint_url
The endpoint URL for an S3-compatible storage service, used for transferring payloads between clients and servers.
```

```{admonition} s3_bucket
The bucket name for an S3-compatible storage service, used for transferring payloads between clients and servers.
```

```{admonition} random_seed
Use a fixed random seed for selecting clients (and sampling testset if needed) so that experiments are reproducible.
```

```{admonition} ping_interval
The time interval in seconds at which the server pings the client.

The default value is 3600.
```

```{admonition} ping_timeout
The time in seconds that the client waits for the server to respond before disconnecting.

The default value is 3600.
```

```{admonition} synchronous
Conduct training in synchronous or asynchronous mode.

The value could be `true` or `false`.
```

```{admonition} periodic_interval
The time interval for a server operating in asynchronous mode to aggregate received updates

Any positive integer could be used for `periodic_interval`. The default value is 5 seconds.
```

```{admonition} simulate_wall_time
Whether the wall clock time on the server is simulated.

The value could be `true` or `false`.
```

```{admonition} staleness_bound
In asynchronous mode, should we wait for stale clients who are behind the current round by more than this bound?

Any positive integer could be used for `staleness_bound`. The default value is 0.
```

```{admonition} minimum_clients_aggregated
The minimum number of clients that need to arrive before aggregation and processing by the server when operating in asynchronous mode. 

Any positive integer could be used for `minimum_clients_aggregated`. The default value is 1.
```

```{admonition} do_test
Whether the central server computes test accuracy locally.

The value could be `true` or `false`.
```

```{admonition} model_path
The directory of pretrained and trained models.

The deafult path is `<base_path>/models/pretrained`.
```

```{admonition} checkpoint_path
The directory of checkpoints.

The default path is `<base_path>/checkpoints`.
```

```{admonition} outbound_processors
A list of processors to apply on the payload before sending. Multiple processor options are available:

- `unstructured_pruning`: Process unstructured pruning on model weights for PyTorch. The `model_compress` processor needs to be applied after it in the configuration file or the communication overhead will not be reduced.

- `structured_pruning`: Process structured pruning on model weights for PyTorch. The `model_compress` processor needs to be applied after it in the configuration file or the communication overhead will not be reduced.

- `model_compress`: Compress model parameters with `Zstandard` compression algorithm. Must be placed as the last processor if applied.
```

```{admonition} inbound_processors
A list of processors to apply on the payload right after receiving. Multiple processor options are available:

- `model_decompress`: Decompress model parameters. Must be placed as the first processor if `model_compress` is applied on the client side.

- `inbound_feature_tensors`: Convert PyTorch tensor features into NumPy arrays before sending to client, for the benefit of saving a substantial amount of communication overhead if the feature dataset is large. Must be used if `clients.outbound_processors` includes `outbound_feature_ndarrays`.

- `feature_dequantize`: Dequantize features for PyTorch MistNet. Must not be used together with `inbound_feature_tensors`.

- `model_dequantize`: Dequantize features for PyTorch model parameters.
```

```{admonition} downlink_bandwidth
Bandwidth for downlink communication (server to clients) in Mbps.

The default value is 100.
```

```{admonition} uplink_bandwidth
Bandwidth for uplink communication (clients to server) in Mbps.

The default value is 100.
```

## data

```{admonition} **dataset**
The training and test datasets. The following options are available:

- `MNIST`
- `FashionMNIST`
- `EMNIST`
- `CIFAR10`
- `CIFAR100`-
- `CINIC10`
- `YOLO`
- `HuggingFace`
- `PASCAL_VOC`
- `TinyImageNet`
- `CelebA`
- `Purchase`
- `Texas`
```

````{admonition} data_path
Where the dataset is located.

The default is `./data`.

```{note}
For the `CINIC10` dataset, the default is `./data/CINIC-10`

For the `TinyImageNet` dataset, the default is `./data/tiny-imagenet-200`
```
````

````{admonition} train_path
Where the training dataset is located.

```{note}
`train_path` need to be specified for datasets using `YOLO`
```
````

````{admonition} test_path
Where the test dataset is located.

```{note}
`test_path` need to be specified for datasets using `YOLO`
```
````

```{admonition} **sampler**
How to divide the entire dataset to the clients. The following options are available:

- `iid`
- `iid_mindspore`
- `noniid`: Could have *concentration* attribute to specify the concentration parameter in the Dirichlet distribution
- `orthogonal`: Each insitution's clients have data of different classes. Could have *institution_class_ids* and *label_distribution* attributes
- `mixed`: Some data are iid, while others are non-iid. Must have *non_iid_clients* attributes
```

````{admonition} test_set_sampler
How to sample the test set when clients test locally. Any **sampler** is valid. 

```{note}
Without this parameter, every client's test set is the test set of the datasource.
```
````

````{admonition} edge_test_set_sampler
How to sample the test set when edge servers test locally. Any **sampler** is valid.

```{note}
Without this parameter, edge servers' test sets are the test set of the datasource if they locally test their aggregated models in cross-silo FL.
```
````

```{admonition} random_seed
Use a fixed random seed to sample each client's dataset so that experiments are reproducible.
```

```{admonition} **partition_size**
Number of samples in each client's dataset. The input value could be any positive integer.
```

```{admonition} concentration
The concentration parameter of symmetric Dirichlet distribution, used by `noniid` **sampler**.

The default value is 1.
```

````{admonition} *non_iid_clients*
Indexs of clients whose datasets are non-iid. Other clients' datasets are iid.

e.g., 4

```{note}
Must have this attribute if the **sampler** is `mixed`
```
````


````{admonition} *institution_class_ids*
Indexs of classes of local data of each institution's clients.

e.g., 0,1;2,3 (the first institution's clients only have data of class #0 and #1; the second institution's clients only have data of class #2 and #3)

```{note}
Could have this attribute if the **sampler** is `orthogonal`
```
````

````{admonition} *label_distribution*
The class distribution of every client's local data.

The value should be `iid` or `noniid`. Default is `iid`

```{note}
Could have this attribute if the **sampler** is `orthogonal`.
```
````

## trainer

```{admonition} **type**
The type of the trainer.

`type` could be set to `basic` or `diff_privacy`.
```

````{admonition} max_physical_batch_size
The limit on the physical batch size when using `diff_privacy` trainer. 

The default value is 128.

```{note}
GPU memory usage of one process training the ResNet-18 model is 2817 MB
```
````

```{admonition} **rounds**
The maximum number of training rounds. 

`round` could be any positive integer.
```

````{admonition} max_concurrency
The maximum number of clients (of each edge server in cross-silo training) running concurrently on one available device. If this is not defined, no new processes are spawned for training.

`max_concurrency` could be any positive integer.

```{note}
Plato will automatically use all available GPUs to maximize the speed of training, launching the same number of clients on every GPU.
```
````

```{admonition} target_accuracy
The target accuracy of the global model.

```

```{admonition} target_perplexity
The target perplexity of the global NLP model.

```

```{admonition} **epochs**
Number of epoches for local training in each communication round.

`epochs` could be any positive integer.
```

```{admonition} **optimizer**
Use `SGD`, `Adam` or `FedProx` for optimizer.

```

```{admonition} **batch_size**
Any positive integer.

```

````{admonition} **learning_rate**

```{note}
Decrease value when using `diff_privacy` trainer.
```
````

```{admonition} **momentum**
```

````{admonition} **weight_decay**

```{note}
When using `diff_privacy` trainer, set to 0.
```
````

```{admonition} lr_schedule
Learning rate scheduler. The following options are available:

- `CosineAnnealingLR`
- `LambdaLR`
- `StepLR`
- `ReduceLROnPlateau`
```

````{admonition} **model_name**
The machine learning model. The following options are available:

- `lenet5`
- `resnet_x`
- `vgg_x`
- `wideresnet`
- `feedback_transformer`
- `yolov5`
- `HuggingFace_CausalLM`
- `inceptionv3`
- `googlenet`
- `unet`
- `alexnet`
- `squeezenet_x`
- `shufflenet_x`
- `dcgan`
- `multilayer`

```{note}
For `resnet_x`, x = 18, 34, 50, 101, or 152

For `vgg_x`, x = 11, 13, 16, or 19

For `squeezenet_x`, x = 0 or 1

For `shufflenet_x`, x = 0.5, 1.0, 1.5, or 2.0
```
````

````{admonition} pretrained
Use a model pretrained on ImageNet or not.

The value for `pretrained ` should be `true` or `false`. Default is `false`.

```{note}
Can be used for `inceptionv3`, `alexnet`, and `squeezenet_x` models.
```
````

```{admonition} dp_epsilon
Total privacy budget of epsilon with the `diff_privacy` trainer.

The default value is 10.0.
```

```{admonition} dp_delta
Total privacy budget of delta with the `diff_privacy` trainer.

The default value is 1e-5.
```

```{admonition} dp_max_grad_norm
The maximum norm of the per-sample gradients with the `diff_privacy` trainer. Any gradient with norm higher than this will be clipped to this value.

The default value is 1.0.
```

```{admonition} num_classes
The number of classes.

The default value is 10.
```


## algorithm

```{admonition} **type**
Aggregation algorithm. 

The input should be:
- `fedavg`:  the federated averaging algorithm
- `mistnet`: the MistNet algorithm

```

````{admonition} *cross_silo*
Cross-silo training. 

The possible input is `true` or `false`.

```{note}
If `true`, must have **total_silos** and **local_rounds** attributes
```
````

```{admonition} *total_silos*
The total number of silos (edge servers). The input could be any positive integer.

```

```{admonition} *local_rounds*
The number of local aggregation rounds on edge servers before sending aggregated weights to the central server. The input could be any positive integer.

```

## results

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
|types|The set of columns that will be written into a .csv file|`round`, `accuracy`, `elapsed_time`, `comm_time`, `round_time`, `comm_overhead`, `local_epoch_num`, `edge_agg_num` (Use comma `,` to seperate them)|default: `round, accuracy, elapsed_time`|
|plot|Plot results |(Format: x\_axis-y\_axis. Use hyphen `-` to seperate axis. Use comma `,` to seperate multiple plots)|default: `round-accuracy, elapsed_time-accuracy`|
|result_path|The directory of results||default: `<base_path>/results/`|