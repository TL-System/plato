## Configuration File

**To be completed**

In Plato, all configuration parameters are read from a configuration file when the clients and the servers launch, and the configuration file follows the YAML format for the sake of simplicity and readability. 

This document introduces all the possible parameters in the configuration file.

Attributes in **bold** must be included in a configuration file, while attributes in *italic* only need to be included under certain conditions.

### clients

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
**type**|Type of federated learning client|`simple`|A basic client who sends weight updates to the server|
|||`mistnet`|A client for MistNet|
|**total_clients**|The total number of clients|A positive number||
|**per_round**|The number of clients selected in each round| Any positive integer that is not larger than **total_clients**||
|**do_test**|Should the clients compute test accuracy locally?| `true` or `false`|| 

### server

| Attribute | Meaning | Valid Values | Note |
|:---------:|:-------:|:-----------:|:----:|
|*type*|The type of the server|`fedavg_cross_silo`|**algorithm.type** must be `fedavg`|
|**address**|The address of the central server|e.g., `127.0.0.1`||
|**port**|The port number of the central server|e.g., `8000`||
|disable_clients|If this optional setting is enabled as `true`, the server will not launched client processes on the same machine.||
|s3_endpoint_url|The endpoint URL for an S3-compatible storage service, used for transferring payloads between clients and servers.||
|s3_bucket|The bucket name for an S3-compatible storage service, used for transferring payloads between clients and servers.||
|ping_interval|The interval in seconds at which the server pings the client. The default is 3600 seconds. |||
|ping_timeout| The time in seconds that the client waits for the server to respond before disconnecting. The default is 360 (seconds).||Increase this number when your session stops running when training larger models (but make sure it is not due to the *out of CUDA memory* error)|
|synchronous|Synchronous or asynchronous federated learning|`true` or `false`|If `false`, must have **periodic_interval** attribute|
|*periodic_interval*|The time interval for a server operating in asynchronous mode to aggregate received updates|Any positive integer||

### data

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
|**dataset**| The training and testing dataset|`MNIST`, `FashionMNIST`, `CIFAR10`, `CINIC10`, `YOLO`, `HuggingFace`, `PASCAL_VOC`, or `TinyImageNet`||
|**data_path**|Where the dataset is located|e.g.,`./data`|For the `CINIC10` dataset, the default `data_path` is `./data/CINIC-10`, For the `TingImageNet` dataset, the default `data_path` is `./data/ting-imagenet-200`|
|**sampler**|How to divide the entire dataset to the clients|`iid`||
|||`iid_mindspore`||
|||`noniid`|Could have *concentration* attribute to specify the concentration parameter in the Dirichlet distribution|
|||`orthogonal`|Each insitution's clients have data of different classes. Could have *institution_class_ids* and *label_distribution* attributes|
|||`mixed`|Some data are iid, while others are non-iid. Must have *non_iid_clients* attributes|
|random_seed|Use a fixed random seed so that experiments are reproducible (clients always have the same datasets)||
|**partition_size**|Number of samples in each client's dataset|Any positive integer||
|concentration| The concentration parameter of symmetric Dirichlet distribution, used by `noniid` **sampler** || Default value is 1|
|*non_iid_clients*|Indexs of clients whose datasets are non-iid. Other clients' datasets are iid|e.g., 4|Must have this attribute if the **sampler** is `mixed`|
|*institution_class_ids*|Indexs of classes of local data of each institution's clients|e.g., 0,1;2,3 (the first institution's clients only have data of class #0 and #1; the second institution's clients only have data of class #2 and #3) |Could have this attribute if the **sampler** is `orthogonal`|
|*label_distribution*|The class distribution of every client's local data|`iid` or `noniid` |Could have this attribute if the **sampler** is `orthogonal`. Default is `iid`|

### trainer

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
|**type**|The type of the trainer|`basic`|
|**rounds**|The maximum number of training rounds|Any positive integer||
|**parallelized**|Whether the training should use multiple GPUs if available|`true` or `false`||
|max_concurrency|The maximum number of clients running concurrently. if this is not defined, no new processes are spawned for training|Any positive integer||
|target_accuracy|The target accuracy of the global model|||
|**epochs**|Number of epoches for local training in each communication round|Any positive integer||
|**optimizer**||`SGD`, `Adam` or `FedProx`||
|**batch_size**||Any positive integer||
|**learning_rate**||||
|**momentum**||||
|**weight_decay**||||   
|lr_schedule|Learning rate scheduler|`CosineAnnealingLR`, `LambdaLR`, `StepLR`, `ReduceLROnPlateau`|| 
|**model_name**|The machine learning model|`lenet5`, `resnet_x`, `vgg_x`,`wideresnet`, `feedback_transformer`, `yolov5`, `HuggingFace_CausalLM`, `inceptionv3`, `googlenet`, `unet`, `alexnet`, `squeezenet_x`, `shufflenet_x`|For `resnet_x`, x = 18, 34, 50, 101, or 152; For `vgg_x`, x = 11, 13, 16, or 19; For `squeezenet_x`, x = 0 or 1; For `shufflenet_x`, x = 0.5, 1.0, 1.5, or 2.0|
|pretrained|Use a model pretrained on ImageNet or not|`true` or `false`. Default is `false`|Can be used for `inceptionv3`, `alexnet`, and `squeezenet_x` models.|

### algorithm

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
|**type**|Aggregation algorithm|`fedavg`|the federated averaging algorithm|
|||`mistnet`|the MistNet algorithm|
|*cross_silo*|Cross-silo training|`true` or `false`|If `true`, must have **total_silos** and **local_rounds** attributes|
|*total_silos*|The total number of silos (edge servers)|Any positive integer||
|*local_rounds*|The number of local aggregation rounds on edge servers before sending aggregated weights to the central server|Any positive integer||

### results

| Attribute | Meaning | Valid Value | Note |
|:---------:|:-------:|:-----------:|:----:|
|types|Which parameter(s) will be written into a CSV file|`accuracy`, `training_time`, `round_time`, `local_epoch_num`, `edge_agg_num`|Use comma `,` to seperate parameters|
|plot|Plot results ||Format: x\_axis&y\_axis. Use comma `,` to seperate multiple plots|
|results_dir|The directory of results||If not specify, results will be stored under `./results/<datasource>/<model>/<server_type>/`|