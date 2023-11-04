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

```{admonition} debug
When `debug` is turned off, the server will try to recover from a failed client by using client processes that are still alive for training. If it's turned on, the server will terminate itself immediately when a client fails.

Valid values are `true` or `false`. The default value is `false`.
```


## clients

```{admonition} **type**
The type of the server.

- `simple` a basic client who sends weight updates to the server.

- `mistnet` a client following the MistNet algorithm.

- `split_learning` a client following the Split Learning algorithm. When this client is used, `clients.do_test` in configuration should be set as `False` because in split learning, we conduct the test on the server.

- `fedavg_personalized` a client saves its local layers before sending the shared global model to the server after local training.

- `self_supervised_learning` a client to prepare the datasource for personalized learning based on self-supervised learning.
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
simulation_distribution:
    distribution: pareto
    alpha: 1
```
````
`````

````{admonition} sleep_simulation
Should clients really go to sleep (`false`), or should we just simulate the sleep times (`true`)? The default is `false`.

Simulating the sleep times — rather than letting clients go to sleep and measure the actual local training times including the sleep times — will be helpful to increase the speed of running the experiments, and to improve reproducibility, since every time the experiments run, the average training time will remain the same, and specified using the `avg_training_time` setting below.

```{admonition} **avg_training_time**
If we are simulating client training times, what is the average training time? When we are simulating the sleep times rather than letting clients go to sleep, we will not be able to use the measured wall-clock time for local training. As a result, we need to specify this value in lieu of the measured training time.
```
````

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

- `model_quantize` Quantize model parameters for PyTorch.

- `model_quantize_qsgd` Quantize model parameters for PyTorch with QSGD.

- `unstructured_pruning` Process unstructured pruning on model weights for PyTorch. The `model_compress` processor needs to be applied after it in the configuration file or the communication overhead will not be reduced.

- `structured_pruning` Process structured pruning on model weights for PyTorch. The `model_compress` processor needs to be applied after it in the configuration file or the communication overhead will not be reduced.

- `model_compress` Compress model parameters with `Zstandard` compression algorithm. Must be placed as the last processor if applied.

- `model_encrypt` Encrypts the model parameters using homomorphic encryption.
```

```{admonition} inbound_processors
A list of processors for the client to apply on the payload before receiving it from the server.

- `model_decompress` Decompress model parameters. Must be placed as the first processor if `model_compress` is applied on the server side.

- `model_decrypt` Decrypts the model parameters using homomorphic encryption.

```

```{admonition} participating_clients_ratio
Percentage of clients participating in federated training out of all clients. The value should range from 0 to 1.
```


## server

```{admonition} type
The type of the server.

- `fedavg` a Federated Averaging (FedAvg) server.

- `fedavg_cross_silo` a Federated Averaging server that handles cross-silo federated learning by interacting with edge servers rather than with clients directly. When this server is used, `algorithm.type` must be `fedavg`.

- `mistnet` a MistNet server.

- `fedavg_gan` a Federated Averaging server that handles Generative Adversarial Networks (GANs).

- `fedavg_he` a Federated Averaging server that handles model updates after homomorphic encryption. When this server is used, the clients need to enable inbound processor `model_decrypt` to decrypt the global model from server, and outbound processor `model_encrypt` to encrypt the model updates.

- `fedavg_personalized` a Federated Averaging server that supports all-purpose personalized federated learning by controlling when and which group of clients are to perform local personalization.

- `split_learning` a Split Learning server that supports training different kinds of models in split learning framework. When this server is used, the `clients.per_round` in the configuration should be set to 1. Users should define the rules for updating models weights before cut from the clients to the server in the callback function `on_update_weights_before_cut`, depending on the specific model they use.

- `fedavg_personalized` a personalized federated learning server that starts from a number of regular
rounds of federated learning. In these regular rounds, only a subset of the total clients can be selected to perform the local update (the ratio of which is a configuration setting). After all regular rounds are completed, it starts a final round of personalization, where a selected subset of clients perform local training using their local dataset.
```

```{admonition} **address**
The address of the central server, such as `127.0.0.1`.
```

```{admonition} **port**
The port number of the central server, such as `8000`.
```

```{admonition} disable_clients
If this optional setting is `true`, the server will not launched client processes on the same physical machine. This is useful when the server is deployed in the cloud and connected to by remote clients.
```

```{admonition} s3_endpoint_url
The endpoint URL for an S3-compatible storage service, used for transferring payloads between clients and servers.
```

```{admonition} s3_bucket
The bucket name for an S3-compatible storage service, used for transferring payloads between clients and servers.
```

```{admonition} random_seed
The random seed used for selecting clients (and sampling the test dataset on the server, if needed) so that experiments are reproducible.
```

```{admonition} ping_interval
The time interval in seconds at which the server pings the client. The default value is `3600`.
```

```{admonition} ping_timeout
The time in seconds that the client waits for the server to respond before disconnecting. The default value is `3600`.
```

```{admonition} synchronous
Whether training session should operate in synchronous (`true`) or asynchronous (`false`) mode.
```

```{admonition} periodic_interval
The time interval for a server operating in asynchronous mode to aggregate received updates. Any positive integer could be used for `periodic_interval`. The default value is 5 seconds. This is only used when we are not simulating the wall-clock time using the `simulate_wall_time` setting below.
```

```{admonition} simulate_wall_time
Whether or not the wall clock time on the server is simulated. This is useful when clients train in batches, rather than concurrently, due to limited resources (such as a limited amount of CUDA memory on the GPUs).
```

```{admonition} staleness_bound
In asynchronous mode, whether or not we should wait for clients who are behind the current round (*stale*) by more than this value. Any positive integer could be used for `staleness_bound`. The default value is `0`.
```

```{admonition} minimum_clients_aggregated
When operating in asynchronous mode, the minimum number of clients that need to arrive before aggregation and processing by the server. Any positive integer could be used for `minimum_clients_aggregated`. The default value is `1`.
```

```{admonition} minimum_edges_aggregated
When operating in asynchronous cross-silo federated learning, the minimum number of edge servers that need to arrive before aggregation and processing by the central server. Any positive integer could be used for `minimum_edges_aggregated`. The default value is `algorithm.total_silos`.
```

```{admonition} do_test
Whether the server tests the global model and computes the global accuracy or perplexity. The default is `true`.
```

```{admonition} model_path
The path to the pretrained and trained models. The default path is `<base_path>/models/pretrained`, where `<base_path>` is specified in the `general` section.
```

```{admonition} checkpoint_path
The path to temporary checkpoints used for resuming the training session. The default path is `<base_path>/checkpoints`, where `<base_path>` is specified in the `general` section.
```

```{admonition} outbound_processors
A list of processors to apply on the payload before sending it out to the clients. Multiple processors are permitted.

- `unstructured_pruning`: Process unstructured pruning on model weights for PyTorch. The `model_compress` processor needs to be applied after it in the configuration file or the communication overhead will not be reduced.

- `structured_pruning`: Process structured pruning on model weights for PyTorch. The `model_compress` processor needs to be applied after it in the configuration file or the communication overhead will not be reduced.

- `model_compress`: Compress model parameters with `Zstandard` compression algorithm. Must be placed as the last processor if applied.
```

```{admonition} inbound_processors
A list of processors to apply on the payload right after receiving. Multiple processors are permitted.

- `model_decompress`: Decompress model parameters. Must be placed as the first processor if `model_compress` is applied on the client side.

- `inbound_feature_tensors`: Convert PyTorch tensor features into NumPy arrays before sending to client, for the benefit of saving a substantial amount of communication overhead if the feature dataset is large. Must be used if `clients.outbound_processors` includes `outbound_feature_ndarrays`.

- `feature_dequantize`: Dequantize features for PyTorch MistNet. Must not be used together with `inbound_feature_tensors`.

- `model_dequantize`: Dequantize PyTorch model parameters back to the 32-bit floating number format.

- `model_dequantize_qsgd`: Dequantize PyTorch model parameters quantized with QSGD.
```

```{admonition} downlink_bandwidth
The server's estimated downlink capacity (server to clients or central server to edge servers in cross-silo training) in Mbps, used for computing the transmission time (see `compute_comm_time` in the `clients` section). The default value is 100.
```

```{admonition} uplink_bandwidth
The server's estimated uplink capacity (server to clients or central server to edge servers in cross-silo training) in Mbps, used for computing the transmission time (see `compute_comm_time` in the `clients` section). The default value is 100.
```

```{admonition} edge_downlink_bandwidth
The edge server's estimated downlink capacity (an edge server to its clients) in Mbps, used for computing the transmission time (see `compute_comm_time` in the `clients` section). The default value is same as `downlink_bandwidth`.
```

```{admonition} edge_uplink_bandwidth
The edge server's estimated uplink capacity (an edge server to its clients) in Mbps, used for computing the transmission time (see `compute_comm_time` in the `clients` section). The default value is same as `uplink_bandwidth`.
```

```{admonition} do_personalization_interval
The round interval for a server commanding when to perform personalization. The default value is 0, meaning that no personalization will be performed.
```

```{admonition} do_personalization_group
The group of clients that is required by the server to perform personalization. There are three options, including "total", "participant", and "nonparticipant". The default value is "participant", meaning the clients participating in training will be used to perform personalization.
```

## data

```{admonition} **dataset**
The training and test datasets. The following options are available:

- `MNIST`
- `FashionMNIST`
- `EMNIST`
- `CIFAR10`
- `CIFAR100`
- `CINIC10`
- `YOLO`
- `HuggingFace`
- `PASCAL_VOC`
- `TinyImageNet`
- `CelebA`
- `Purchase`
- `Texas`
- `STL10`
```

````{admonition} data_path
Where the dataset is located. The default is `./data`.

```{note}
For the `CINIC10` dataset, the default is `./data/CINIC-10`

For the `TinyImageNet` dataset, the default is `./data/tiny-imagenet-200`
```
````

````{admonition} train_path
Where the training dataset is located.

```{note}
`train_path` need to be specified for datasets using `YOLO`.
```
````

````{admonition} test_path
Where the test dataset is located.

```{note}
`test_path` need to be specified for datasets using `YOLO`.
```
````

````{admonition} **sampler**
How to divide the entire dataset to the clients. The following options are available:

- `iid`

- `iid_mindspore`

- `noniid`: Could have *concentration* attribute to specify the concentration parameter in the Dirichlet distribution

```{admonition} concentration
If the sampler is `noniid`, the concentration parameter for the Dirichlet distribution can be specified. The default value is `1`.
```

- `orthogonal`: Each institution's clients have data of different classes. Could have *institution_class_ids* and *label_distribution* attributes

```{admonition} institution_class_ids
If the sampler is `orthogonal`, the indices of classes of local data of each institution's clients can be specified. e.g., `0, 1; 2, 3` (the first institution's clients only have data of class #0 and #1; the second institution's clients only have data of class #2 and #3).
```

```{admonition} label_distribution
If the sampler is `orthogonal`, the class distribution of every client's local data can be specified. The value should be `iid` or `noniid`. Default is `iid`.
```

- `mixed`: Some data are iid, while others are non-iid. Must have *non_iid_clients* attributes

```{admonition} non_iid_clients
If the sampler is `mixed`, the indices of clients whose datasets are non-i.i.d. need to be specified. Other clients' datasets are i.i.d.
```
````

````{admonition} testset_sampler
How the test dataset is sampled when clients test locally. Any sampler type is valid.

```{note}
Without this parameter, the test dataset on either the client or the server is the entire test dataset of the datasource.
```
````

```{admonition} random_seed
The random seed used to sample each client's dataset so that experiments are reproducible.
```

```{admonition} **partition_size**
The number of samples in each client's dataset.
```

```{admonition} testset_size
The number of samples in the server's test dataset when server-side evaluation is conducted; PyTorch only (for now).
```

## trainer

````{admonition} **type**
The type of the trainer. The following types are available:
- `basic`: a basic trainer with a standard training loop.
- `diff_privacy`: a trainer that supports local differential privacy in its training loop by adding noise to the gradients during each step of training.
- `split_learning`: a trainer that supports the split learning framework.
- `self_supervised_learning`: a trainer that supports personalized federated learning based on self supervised learning.

```{admonition} max_physical_batch_size
The limit on the physical batch size when using the `diff_privacy` trainer.  The default value is 128. The GPU memory usage of one process training the ResNet-18 model is around 2817 MB.
```

```{admonition} dp_epsilon
Total privacy budget of epsilon with the `diff_privacy` trainer. The default value is `10.0`.
```

```{admonition} dp_delta
Total privacy budget of delta with the `diff_privacy` trainer. The default value is `1e-5`.
```

```{admonition} dp_max_grad_norm
The maximum norm of the per-sample gradients with the `diff_privacy` trainer. Any gradient with norm higher than this will be clipped to this value. The default value is `1.0`.
```

- `gan`: a trainer for Generative Adversarial Networks (GANs).
````


```{admonition} **rounds**
The maximum number of training rounds.

`round` could be any positive integer.
```

````{admonition} max_concurrency
The maximum number of clients (of each edge server in cross-silo training) running concurrently on each available GPU. If this is not defined, no new processes are spawned for training.

```{note}
Plato will automatically use all available GPUs to maximize the concurrency of training, launching the same number of clients on every GPU. If `max_concurrency` is 7 and 3 GPUs are available, 21 client processes will be launched for concurrent training.
```
````

```{admonition} target_accuracy
The target accuracy of the global model.
```

```{admonition} target_perplexity
The target perplexity of the global Natural Language Processing (NLP) model.
```

```{admonition} **epochs**
The total number of epochs in local training in each communication round.
```

```{admonition} **batch_size**
The size of the mini-batch of data in each step (iteration) of the training loop.
```

```{admonition} **optimizer**
The type of the optimizer. The following options are supported:

- `Adam`
- `Adadelta`
- `Adagrad`
- `AdaHessian` (from the `torch_optimizer` package)
- `AdamW`
- `SparseAdam`
- `Adamax`
- `ASGD`
- `LBFGS`
- `NAdam`
- `RAdam`
- `RMSprop`
- `Rprop`
- `SGD`
```

````{admonition} lr_scheduler
The learning rate scheduler. The following learning rate schedulers from PyTorch are supported:

- `CosineAnnealingLR`
- `LambdaLR`
- `MultiStepLR`
- `StepLR`
- `ReduceLROnPlateau`
- `ConstantLR`
- `LinearLR`
- `ExponentialLR`
- `CyclicLR`
- `CosineAnnealingWarmRestarts`

Alternatively, all four schedulers from [timm](https://timm.fast.ai/schedulers) are supported if `lr_scheduler` is specified as `timm` and `trainer -> type` is specified as `timm_basic`. For example, to use the `SGDR` scheduler, we specify `cosine` as `sched` in its arguments (`parameters -> learning_rate`):

```
trainer:
    type: timm_basic

parameters:
    learning_rate:
        sched: cosine
        min_lr: 1.e-6
        warmup_lr: 0.0001
        warmup_epochs: 3
        cooldown_epochs: 10
```

````

```{admonition} loss_criterion
The loss criterion. The following options are supported:

- `L1Loss`
- `MSELoss`
- `BCELoss`
- `BCEWithLogitsLoss`
- `NLLLoss`
- `PoissonNLLLoss`
- `CrossEntropyLoss`
- `HingeEmbeddingLoss`
- `MarginRankingLoss`
- `TripletMarginLoss`
- `KLDivLoss`
- `NegativeCosineSimilarity`
- `NTXentLoss`
- `SwaVLoss`

```

```{admonition} global_lr_scheduler
Whether the learning rate should be scheduled globally (`true`) or not (`false`).
If `true`, the learning rate of the first epoch in the next communication round is scheduled based on that of the last epoch in the previous communication round.
```

````{admonition} **model_type**
The repository where the machine learning model should be retrieved from. The following options are available:

- `cnn_encoder` (for generating various encoders by extracting from CNN models such as ResNet models)
- `general_multilayer` (for generating a multi-layer perceptron using a provided configuration)
- `huggingface` (for [HuggingFace](https://huggingface.co/models) causal language models)
- `torch_hub` (for models from [PyTorch Hub](https://pytorch.org/hub/))
- `vit` (for Vision Transformer models from [HuggingFace](https://huggingface.co/models), [Tokens-to-Token ViT](https://github.com/yitu-opensource/T2T-ViT), and [Deep Vision Transformer](https://github.com/zhoudaquan/dvit_repo))

The name of the model should be specified below, in `model_name`.

````{note}
For `vit`, please replace the `/` in model name from [https://huggingface.co/models](https://huggingface.co/models) with `@`. For example, use `google@vit-base-patch16-224-in21k` instead of `google/vit-base-patch16-224-in21k`. If you do not want to use the pretrained weights, set `parameters -> model -> pretrained` to `false`, as in the following example:

```
parameters:
    model:
        pretrained: false
```
````

````{admonition} **model_name**
The name of the machine learning model. The following options are available:

- `lenet5`
- `resnet_x`
- `vgg_x`
- `yolov5`
- `dcgan`
- `multilayer`

```{note}
If the `model_type` above specified a model repository, supply the name of the model, such as `gpt2`, here.

For `resnet_x`, x = 18, 34, 50, 101, or 152; For `vgg_x`, x = 11, 13, 16, or 19.
```
````


## algorithm

```{admonition} **type**
Aggregation algorithm.

The input should be:
- `fedavg`:  the federated averaging algorithm
- `mistnet`: the MistNet algorithm
- `split_learning`: the Split Learning algorithm
- `fedavg_personalized`: the personalized federated learning algorithm
```

````{admonition} cross_silo
Whether or not cross-silo training should be used.

```{admonition} **total_silos**
The total number of silos (edge servers). The input could be any positive integer.
```

```{admonition} **local_rounds**
The number of local aggregation rounds on edge servers before sending aggregated weights to the central server. The input could be any positive integer.
```
````

```{admonition} **local_rounds**
The number of local aggregation rounds on edge servers before sending aggregated weights to the central server. The input could be any positive integer.
```
````

````{admonition} fedavg_personalized
Whether or not the personalized training should be used. 

```{admonition} **local_layer_names**
Local layers in a model should remain local at the clients during personalized FL training, and should not be aggregated at the server.

```
```{admonition} **participating_clients_ratio**
A float to show the proportion of clients participating in the federated training process.  It is under `personalization`, which is a sub-config path that contains other personalized training parameters. Default: 1.0
```
````

## results

````{admonition} types
The set of columns that will be written into a .csv file.

The valid values are:
- `round`
- `accuracy`
- `elapsed_time`
- `comm_time`
- `processing_time`
- `round_time`
- `comm_overhead`
- `local_epoch_num`
- `edge_agg_num`

```{note}
Use comma `,` to separate them. The default is `round, accuracy, elapsed_time`.
```
````

````{admonition} result_path
The path to the result `.csv` files. The default path is `<base_path>/results/`,  where `<base_path>` is specified in the `general` section.
````

## parameters

````{note}
Your parameters in your configuration file must match the keywords in `__init__` of your model, optimizer, learning rate scheduler, or loss criterion. For example, if you want to set `base_lr` in the learning scheduler `CyclicLR`, you will need:

```
parameters:
    learning_rate:
        base_lr: 0.01
```
````

```{admonition} model
All the parameter settings that need to be passed as keyword parameters when initializing the model, such as `num_classes` or `cut_layer`. The set of parameters permitted or needed depends on the model.
```

```{admonition} optimizer
All the parameter settings that need to be passed as keyword parameters when initializing the optimizer, such as `lr`, `momentum`, or `weight_decay`. The set of parameters permitted or needed depends on the optimizer.
```

```{admonition} learning_rate
All the parameter settings that need to be passed as keyword parameters when initializing the learning rate scheduler, such as `gamma`. The set of parameters permitted or needed depends on the learning rate scheduler.
```

```{admonition} loss_criterion
All the parameter settings that need to be passed as keyword parameters when initializing the loss criterion, such as `size_average`. The set of parameters permitted or needed depends on the loss criterion.
```
