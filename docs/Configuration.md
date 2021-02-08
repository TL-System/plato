
## Configuration File

**To be completed**

In Plato, all configuration parameters are read from a configuration file when the clients and the servers launch, and the configuration file follows the YAML format for the sake of simplicity and readability. 

This document introduces all the possible parameters in the configuration file.

Attributes in **bold** are must included in a configuration file, in *italic* are must included under certain conditions.

### clients

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
**type**|Type of federated learning client|`simple`|A basic client who sends weight updates to the server|
|||`mistnet`|A client for MistNet|
|**total_clients**|The total number of clients| Any positive integer||
|**per_round**|The number of clients selected in each round| Any positive integer that is not larger than **total_clients**||
|**do_test**|Should the clients compute test accuracy locally?| `true` or `false`|| 

### server

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
|**address**|The address of the central server|e.g., `localhost`||
|**port**|The port number of the central server|e.g., `8000`||

### data

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
|**dataset**| The training and testing dataset|`MNIST`, `FashionMNIST`, `CIFAR10`, or `CINIC10`||
|**data_path**|Where the dataset is located|e.g.,`./data`||
|**divider**|How to divide the whole dataset on clients|`iid`|Must have *partition_size* attribute|
|||`iid_mindspore`|Must have *partition_size* attribute|
|||`bias`|Must have *partition_size* and *label_distribution* attributes|
|||`shard`|First sort the whole dataset by labels, then divide into (*shard_per\_client* x **clients.total_clients**) shards of equal size, and assign each client **shard_per\_client** shard(s)|
|*partition_size*|Number of samples in each partition|Any positive integer|Must have this attribute if **divider** is `iid`, `iid_mindspore` or `bias`|
|*label_distribution*|Uniform or normal distribution for labels across clients?|`uniform` or `normal`|Must have this attribute if **divider** is `bias`|
|*shard_per\_client*|Number of shards per client|Any positive integer|Must have this attribute if **divider** is `shard`|


### trainer

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
|**type**|The type of the trainer|`basic`|
|**rounds**|The maximum number of training rounds|Any positive integer||
|**parallelized**|Whether the training should use multiple GPUs if available|`true` or `false`||
|**max_concurrency**|The maximum number of clients running concurrently|Any positive integer||
|target_accuracy|The target accuracy of the global model|||
|**epochs**|Number of epoches for local training in each communication round|Any positive integer||
|**optimizer**||`SGD`, `Adam` or `FedProx`||
|**batch_size**||Any positive integer||
|**learning_rate**||||
|**momentum**||||
|**weight_decay**||||    
|**model**|The machine learning model|`lenet5`, `resnet`, `wideresnet`, `vgg`||

### algorithm

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
|**type**|Aggregation algorithm|`fedavg`|
|||`fedavg_cross_silo`||
|||`mistnet`||
|||`adaptive_sync`||
|*cross_silo*|Cross-silo training|`true` or `false`|If `true`, must have **total_silos** and **local_rounds** attributes|
|*total_silos*|The total number of silos (edge servers)|Any positive integer||
|*local_rounds*|The number of local aggregation rounds on edge servers before sending aggregated weights to the central server|Any positive integer||

### results

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
|types|Which parameter(s) will be written into a CSV file|`accuracy`, `training_time`, `round_time`, `local_epoch_num`, `edge_agg_num`|Use comma `,` to seperate parameters|
|plot|Plot results ||Format: x\_axis&y\_axis|