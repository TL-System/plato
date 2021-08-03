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
|*type*|The type of the server|`fedavg_cross_silo`|**algorithm.type** must be `fedavg`|
|**address**|The address of the central server|e.g., `127.0.0.1`||
|**port**|The port number of the central server|e.g., `8000`||

### data

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
|**dataset**| The training and testing dataset|`MNIST`, `FashionMNIST`, `CIFAR10`, `CINIC10`, or `COCO`||
|**data_path**|Where the dataset is located|e.g.,`./data`||
|**sampler**|How to divide the entire dataset to the clients|`iid`||
|||`iid_mindspore`||
|||`noniid`|Could have *concentration* attribute to specify the concentration parameter in the Dirichlet distribution|
|||`mixed`|Some clients' datasets are iid. Some are non-iid. Must have *non_iid_clients* attributes|
|random_seed|Keep a random seed to make experiments reproducible (clients always have the same datasets)||
|**partition_size**|Number of samples in each client's dataset|Any positive integer||
|*non_iid_clients*|Indexs of clients whose datasets are non-iid. Other clients' datasets are iid|e.g., 4|Must have this attribute if the **sampler** is `mixed`|



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
|**model_name**|The machine learning model|`lenet5`, `resnet`, `vgg`,`wideresnet`, `feedback_transformer`, `yolov5`, `HuggingFace_CausalLM`||

### algorithm

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
|**type**|Aggregation algorithm|`fedavg`|
|||`mistnet`||
|||`adaptive_sync`||
|||`adaptive_freezing`||
|*cross_silo*|Cross-silo training|`true` or `false`|If `true`, must have **total_silos** and **local_rounds** attributes|
|*total_silos*|The total number of silos (edge servers)|Any positive integer||
|*local_rounds*|The number of local aggregation rounds on edge servers before sending aggregated weights to the central server|Any positive integer||

### results

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
|types|Which parameter(s) will be written into a CSV file|`accuracy`, `training_time`, `round_time`, `local_epoch_num`, `edge_agg_num`|Use comma `,` to seperate parameters|
|plot|Plot results ||Format: x\_axis&y\_axis|
|results_dir|The directory of results||If not specify, results will be stored under `./results/<datasource>/<model>/<server_type>/`|
|trainer_counter_dir|The directory of running_trainers.sqlitedb||If not specify, it will be stored under `__file__`|