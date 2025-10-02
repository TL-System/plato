# Examples

In `examples/`, we included a wide variety of examples that showcased how third-party deep learning frameworks, such as [Catalyst](https://catalyst-team.github.io/catalyst/), can be used, and how a collection of federated learning algorithms in the research literature can be implemented using Plato by customizing the `client`, `server`, `algorithm`, and `trainer`. We also included detailed tutorials on how Plato can be run on Google Colab. Here is a list of the examples we included.

### Before Starting:
#### Downloading the Dataset
If you have not yet downloaded the dataset, run the command with the "-d" flag. Once you see the following message:

```shell
The dataset has been successfully downloaded. Re-run the experiment without '-d' or '--download'.
```

re-run the command without the "-d" flag.

For example:
```shell
uv run examples/personalized_fl/fedbabu/fedbabu.py -c examples/personalized_fl/configs/fedbabu_CIFAR10_resnet18.yml -d
uv run examples/personalized_fl/fedbabu/fedbabu.py -c examples/personalized_fl/configs/fedbabu_CIFAR10_resnet18.yml
```
#### Dependencies
Plato uses "uv" to organize dependencies in a hierarchical manner. Packages required only for specific features are defined in the 'pyproject.toml' file within their respective directories, rather than in the root directory. To run an example with its specific dependencies, navigate to the corresponding example folder and execute "uv run". For example:
```
cd examples/ssl/smog
uv run smog.py -c ../../../examples/ssl/configs/smog_CIFAR10_resnet18.yml
```

#### Support for Third-Party Frameworks

#### Server Aggregation Algorithms

````{admonition} **FedAtt**
FedAtt is a server aggregation algorithm, where client updates were aggregated using a layer-wise attention-based mechanism that considered the similarity between the server and client models.  The objective was to improve the accuracy or perplexity of the trained model with the same number of communication rounds. In its implementation in `examples/fedatt/fedatt_algorithm.py`, the PyTorch implementation of FedAtt overrides `aggregate_weights()` to implement FedAtt as a custom server aggregation algorithm.

```shell
uv run examples/server_aggregation/fedatt/fedatt.py -c examples/server_aggregation/fedatt/fedatt_FashionMNIST_lenet5.yml
```

```{note}
S. Ji, S. Pan, G. Long, X. Li, J. Jiang, Z. Huang. &ldquo;[Learning Private Neural Language Modeling with Attentive Aggregation](https://arxiv.org/abs/1812.07108),&rdquo; in Proc. International Joint Conference on Neural Networks (IJCNN), 2019.
```
````

````{admonition} **FedAdp**
FedAdp is another server aggregation algorithm, which exploited the implicit connection between data distribution on a client and the contribution from that client to the global model, measured at the server by inferring gradient information of participating clients. In its implementation in `examples/fedadp/fedadp_server.py`, a framework-agnostic implementation of FedAdp overrides `aggregate_deltas()` to implement FedAdp as a custom server aggregation algorithm.

```shell
uv run examples/server_aggregation/fedadp/fedadp.py -c examples/server_aggregation/fedadp/fedadp_FashionMNIST_lenet5.yml
```

```{note}
H. Wu, P. Wang. &ldquo;[Fast-Convergent Federated Learning with Adaptive Weighting](https://ieeexplore.ieee.org/abstract/document/9442814),&rdquo; in IEEE Trans. on Cognitive Communications and Networking (TCCN), 2021.
```
````

#### Secure Aggregation with Homomorphic Encryption

````{admonition} **MaskCrypt**
MaskCrypt is a secure federated learning system based on homomorphic encryption. Instead of encrypting all the model updates, MaskCrypt encrypts only part of them to balance the tradeoff between security and efficiency. In this example, clients only select 5% of the model updates to encrypt during the learning process. The number of encrypted weights is determined by `encrypt_ratio`, which can be adjusted in the configuration file. A random mask will be adopted if `random_mask` is set to true.

```shell
uv run examples/secure_aggregation/maskcrypt/maskcrypt.py -c examples/secure_aggregation/maskcrypt/maskcrypt_MNIST_lenet5.yml
```
````

#### Asynchronous Federated Learning Algorithms

````{admonition} **FedAsync**
FedAsync is one of the first algorithms proposed in the literature towards operating federated learning training sessions in *asynchronous* mode, which Plato supports natively. It advocated aggregating aggressively whenever only *one* client reported its local updates to the server.

In its implementation, FedAsync's server subclasses from the `FedAvg` server and overrides its `configure()` and `aggregate_weights()` functions. In `configure()`, it needs to add some custom features (of obtaining a mixing hyperparameter for later use in the aggregation process), and calls `super().configure()` first, similar to its `__init__()` function calling `super().__init__()`. When it overrides `aggregate_weights()`, however, it supplied a completely custom implementation of this function.

```shell
uv run examples/async/fedasync/fedasync.py -c examples/async/fedasync/fedasync_MNIST_lenet5.yml
```

```{note}
C. Xie, S. Koyejo, I. Gupta. &ldquo;[Asynchronous Federated Optimization](https://opt-ml.org/papers/2020/paper_28.pdf),&rdquo; in Proc. Annual Workshop on Optimization for Machine Learning (OPT), 2020.
```
````

````{admonition} **Port**
Port is one of the federated learning training sessions in *asynchronous* mode. The server will aggregate when it receives a minimum number of clients' updates, which can be tuned with 'minimum_clients_aggregated'. The 'staleness_bound' is also a common parameter in asynchronous FL, which limit the staleness of all clients' updates. 'request_update' is a special design in Port, to force clients report their updates and shut down the training process if their too slow. 'similarity_weight' and 'staleness_weight' are two hyper-parameters in Port, tuning the weights of them when the server do aggregation. 'max_sleep_time', 'sleep_simulation', 'avg_training_time' and 'simulation_distribution' are also important to define the arrival clients in Port.

```shell
uv run examples/async/port/port.py -c examples/async/port/port_cifar10.yml
```

```{note}
N. Su, B. Li. &ldquo;[How Asynchronous can Federated Learning Be?](https://ieeexplore.ieee.org/document/9812885),&rdquo; in Proc. IEEE/ACM International Symposium on Quality of Service (IWQoS), 2022.
```
````

#### Federated Unlearning

````{admonition} **Federated Unlearning**
Federated unlearning is a concept proposed in the recent research literature that uses an unlearning algorithm, such as retraining from scratch, to guarantee that a client is able to remove all the effects of its local private data samples from the trained model.  In its implementation in `examples/unlearning/fedunlearning/fedunlearning_server.py` and `examples/unlearning/fedunlearning/fedunlearning_client.py`, a framework-agnostic implementation of federated unlearning overrides several methods in the client and server APIs, such as the server's `aggregate_deltas()` to implement federated unlearning.

```shell
uv run examples/unlearning/fedunlearning/fedunlearning.py -c examples/unlearning/fedunlearning/fedunlearning_adahessian_MNIST_lenet5.yml
```

```{note}
If the AdaHessian optimizer is used as in the example configuration file, it will reflect what the following paper proposed:

Liu et al., &ldquo;[The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid Retraining](https://arxiv.org/abs/2203.07320),&rdquo; in Proc. INFOCOM, 2022.
```
````

````{admonition} **Knot**
Knot is implemented in `examples/unlearning`, which clusters the clients, and the server aggregation is carried out within each cluster only. Knot is designed under **asynchronous** mode, and unlearned by retraining from scratch in cluster. The global model will be aggregated at the end of the retraining process. Knot supports a wide range of tasks, including image classification and language tasks.

```shell
uv run examples/unlearning/knot/knot.py -c examples/unlearning/knot/knot_cifar10_resnet18.yml
uv run examples/unlearning/knot/knot.py -c examples/unlearning/knot/knot_mnist_lenet5.yml
uv run examples/unlearning/knot/knot.py -c examples/unlearning/knot/knot_purchase.yml
```

```{note}
N. Su, B. Li. &ldquo;[Asynchronous Federated Unlearning](https://iqua.ece.toronto.edu/papers/ningxinsu-infocom23.pdf),&rdquo; in Proc. IEEE International Conference on Computer Communications (INFOCOM 2023).
```
````

#### Algorithms with Customized Client Training Loops

````{admonition} **SCAFFOLD**
SCAFFOLD is a synchronous federated learning algorithm that performs server aggregation with control variates to better handle statistical heterogeneity. It has been quite widely cited and compared with in the federated learning literature. In this example, two processors, called `ExtractControlVariatesProcessor` and `SendControlVariateProcessor`, have been introduced to the client using a callback class, called `ScaffoldCallback`. They are used for sending control variates between the clients and the server. Each client also tries to maintain its own control variates for local optimization using files.

```shell
uv run examples/customized_client_training/scaffold/scaffold.py -c examples/customized_client_training/scaffold/scaffold_MNIST_lenet5.yml
```

```{note}
Karimireddy et al., &ldquo;[SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a.html), &rdquo; in Proc. International Conference on Machine Learning (ICML), 2020.
```
````

````{admonition} **FedProx**
To better handle system heterogeneity, the FedProx algorithm introduced a proximal term in the optimizer used by local training on the clients. It has been quite widely cited and compared with in the federated learning literature.

```shell
uv run examples/customized_client_training/fedprox/fedprox.py -c examples/customized_client_training/fedprox/fedprox_MNIST_lenet5.yml
```

```{note}
T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith. &ldquo;[Federated Optimization in Heterogeneous Networks](https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf),&rdquo; in Proc. Machine Learning and Systems (MLSys), 2020.
```
````

````{admonition} **FedDyn**
FedDyn is proposed to provide communication savings by dynamically updating each participating device's regularizer in each round of training. It is a method proposed to solve data heterogeneity in federated learning.
```shell
uv run examples/customized_client_training/feddyn/feddyn.py -c examples/customized_client_training/feddyn/feddyn_MNIST_lenet5.yml
```

```{note}
Acar, D.A.E., Zhao, Y., Navarro, R.M., Mattina, M., Whatmough, P.N. and Saligrama, V. &ldquo;[Federated learning based on dynamic regularization](https://openreview.net/forum?id=B7v4QMR6Z9w),&rdquo; Proceedings of International Conference on Learning Representations (ICLR), 2021.
```
````


````{admonition} **FedMos**
FedMoS is a communication-efficient FL framework with coupled double momentum-based update and adaptive client selection, to jointly mitigate the intrinsic variance.

```shell
uv run examples/customized_client_training/fedmos/fedmos.py -c examples/customized_client_training/fedmos/fedmos_MNIST_lenet5.yml
```
```{note}
X. Wang, Y. Chen, Y. Li, X. Liao, H. Jin and B. Li, "FedMoS: Taming Client Drift in Federated Learning with Double Momentum and Adaptive Selection," IEEE INFOCOM 2023.
````
#### Client Selection Algorithms

````{admonition} **Active Federated Learning**
Active Federated Learning is a client selection algorithm, where clients were selected not uniformly at random in each round, but with a probability conditioned on the current model and the data on the client to maximize training efficiency. The objective was to reduce the number of required training iterations while maintaining the same model accuracy. In its implementation in `examples/afl/afl_server.py`, the server overrides `choose_clients()` to implement a custom client selection algorithm, and overrides `weights_aggregated()` to extract additional information from client reports.

```shell
uv run examples/client_selection/afl/afl.py -c examples/client_selection/afl/afl_FashionMNIST_lenet5.yml
```

```{note}
J. Goetz, K. Malik, D. Bui, S. Moon, H. Liu, A. Kumar. &ldquo;[Active Federated Learning](https://arxiv.org/abs/1909.12641),&rdquo; September 2019.
```
````

````{admonition} **Pisces**
Pisces is an asynchronous federated learning algorithm that performs biased client selection based on overall utilities and weighted server aggregation based on staleness. In this example, a client running the Pisces algorithm calculates its statistical utility and report it together with model updates to Pisces server. The server then evaluates the overall utility for each client based on the reported statistical utility and client staleness, and selects clients for the next communication round. The algorithm also attempts to detect outliers via DBSCAN for better robustness.

```shell
uv run examples/client_selection/pisces/pisces.py -c examples/client_selection/pisces/pisces_MNIST_lenet5.yml
```

```{note}
Jiang et al., &ldquo;[Pisces: Efficient Federated Learning via Guided Asynchronous Training](https://arxiv.org/pdf/2206.09264.pdf), &rdquo; in Proc. ACM Symposium on Cloud Computing (SoCC), 2022.
```
````

````{admonition} **Oort**
Oort is a federated learning algorithm that performs biased client selection based on both statistical utility and system utility. Originally, Oort is proposed for synchronous federated learning. In this example, it was adapted to support both synchronous and asynchronous federated learning. Notably, the Oort server maintains a blacklist for clients that have been selected too many times (10 by default). If `per_round` / `total_clients` is large, e.g. 2/5, the Oort server may not work correctly because most clients are in the blacklist and there will not be a sufficient number of clients that can be selected.

```shell
uv run examples/client_selection/oort/oort.py -c examples/client_selection/oort/oort_MNIST_lenet5.yml
```

```{note}
Lai et al., &ldquo;[Oort: Efficient Federated Learning via Guided Participant Selection](https://www.usenix.org/system/files/osdi21-lai.pdf),&rdquo; in Proc. USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2021.
```
````
````{admonition} **Polaris**
Polaris is a client selection method for asynchronous federated learning. In this method, it selects clients via balancing between local device speed and local data quality from an optimization perspective. As it does not require extra information rather than local updates, Polaris is pluggable to any other federated aggregation methods.

```shell
uv run examples/client_selection/polaris/polaris.py -c examples/client_selection/polaris/polaris_FEMNIST_LeNet5.yml
```

```{note}
Kang et al., &ldquo;[POLARIS: Accelerating Asynchronous Federated Learning with Client Selection],
&rdquo;
````

#### Split Learning Algorithms

````{admonition} **Split Learning**
Split learning aims to collaboratively train deep learning models with the server performing a portion of the training process. In split learning, each training iteration is separated into two phases: the clients first send extracted features at a specific cut layer to the server, and then the server continues the forward pass and computes gradients, which will be sent back to the clients to complete the backward pass of the training. Unlike federated learning, split learning clients sequentially interact with the server, and the global model is synchronized implicitly through the model on the server side, which is shared and updated by all clients.
```shell
uv run plato.py -c configs/CIFAR10/split_learning_resnet18.yml
```

```{note}
Vepakomma et al., &ldquo;[Split Learning for Health: Distributed Deep Learning without Sharing Raw Patient Data](https://arxiv.org/abs/1812.00564),&rdquo; in Proc. NeurIPS, 2018.
```
````

````{admonition} **Split Learning for Training LLM**
This is an example of fine-tuning the Hugging Face large language model with split learning. The fine-tuning policy includes training the whole model and fine-tuning with the LoRA algorithm. The cut layer in the configuration file should be set as an integer, indicating cutting at which transformer block in the transformer model.

Fine-tune the whole model
```shell
uv run ./examples/split_learning/llm_split_learning/split_learning_main.py -c ./examples/split_learning/llm_split_learning/split_learning_wikitext103_gpt2.yml
```
Fine-tune with LoRA
```shell
uv run ./examples/split_learning/llm_split_learning/split_learning_main.py -c ./examples/split_learning/llm_split_learning/split_learning_wikitext2_gpt2_lora.yml
```
````

#### Personalized Federated Learning Algorithms

````{admonition} **FedRep**
FedRep learns a shared data representation (the global layers) across clients and a unique, personalized local ``head'' (the local layers) for each client. In this implementation, after each round of local training, only the representation on each client is retrieved and uploaded to the server for aggregation.

```shell
uv run examples/personalized_fl/fedrep/fedrep.py -c examples/personalized_fl/configs/fedrep_CIFAR10_resnet18.yml
```

```{note}
Collins et al., &ldquo;[Exploiting Shared Representations for Personalized Federated Learning](http://proceedings.mlr.press/v139/collins21a/collins21a.pdf), &rdquo; in Proc. International Conference on Machine Learning (ICML), 2021.
```
````

````{admonition} **FedBABU**
FedBABU only updates the global layers of the model during FL training. The local layers are frozen at the beginning of each local training epoch.

```shell
uv run examples/personalized_fl/fedbabu/fedbabu.py -c examples/personalized_fl/configs/fedbabu_CIFAR10_resnet18.yml
```

```{note}
Oh et al., &ldquo;[FedBABU: Towards Enhanced Representation for Federated Image Classification](https://openreview.net/forum?id=HuaYQfggn5u),
&rdquo; in Proc. International Conference on Learning Representations (ICLR), 2022.
```
````

````{admonition} **APFL**
APFL jointly optimizes the global model and personalized models by interpolating between local and personalized models. Once the global model is received, each client will carry out a regular local update, and then conduct a personalized optimization to acquire a trained personalized model. The trained global model and the personalized model will subsequently be combined using the parameter "alpha," which can be dynamically updated.

```shell
uv run examples/personalized_fl/apfl/apfl.py -c examples/personalized_fl/configs/apfl_CIFAR10_resnet18.yml
```

```{note}
Deng et al., &ldquo;[Adaptive Personalized Federated Learning](https://arxiv.org/abs/2003.13461),
&rdquo; in Arxiv, 2021.
```
````

````{admonition} **FedPer**
FedPer learns a global representation and personalized heads, but makes simultaneous local updates for both sets of parameters, therefore makes the same number of local updates for the head and the representation on each local round.

```shell
uv run examples/personalized_fl/fedper/fedper.py -c examples/personalized_fl/configs/fedper_CIFAR10_resnet18.yml
```

```{note}
Arivazhagan et al., &ldquo;[Federated learning with personalization layers](https://arxiv.org/abs/1912.00818), &rdquo; in Arxiv, 2019.
```
````

````{admonition} **LG-FedAvg**
With LG-FedAvg only the global layers of a model are sent to the server for aggregation, while each client keeps local layers to itself.

```shell
uv run examples/personalized_fl/lgfedavg/lgfedavg.py -c examples/personalized_fl/configs/lgfedavg_CIFAR10_resnet18.yml
```

```{note}
Liang et al., &ldquo;[Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523), &rdquo; in Proc. NeurIPS, 2019.
```
````

````{admonition} **Ditto**
Ditto jointly optimizes the global model and personalized models by learning local models that are encouraged to be close together by global regularization. In this example, once the global model is received, each client will carry out a regular local update and then optimizes the personalized model.

```shell
uv run examples/personalized_fl/ditto/ditto.py -c examples/personalized_fl/configs/ditto_CIFAR10_resnet18.yml
```

```{note}
Li et al., &ldquo;[Ditto: Fair and robust federated learning through personalization](https://proceedings.mlr.press/v139/li21h.html), &rdquo; in Proc ICML, 2021.
```
````

````{admonition} **Per-FedAvg**
Per-FedAvg uses the Model-Agnostic Meta-Learning (MAML) framework to perform local training during the regular training rounds. It performs two forward and backward passes with fixed learning rates in each iteration.

```shell
uv run examples/personalized_fl/perfedavg/perfedavg.py -c examples/personalized_fl/configs/perfedavg_CIFAR10_resnet18.yml
```

```{note}
Fallah et al., &ldquo;[Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html), &rdquo; in Proc NeurIPS, 2020.
```
````

````{admonition} Hermes

Hermes utilizes structured pruning to improve both communication efficiency and inference efficiency of federated learning. It prunes channels with the lowest magnitudes in each local model and adjusts the pruning amount based on each local model’s test accuracy and its previous pruning amount. When the server aggregates pruned updates, it only averages parameters that were not pruned on all clients.

```shell

uv run examples/personalized_fl/hermes/hermes.py -c examples/personalized_fl/configs/hermes_CIFAR10_resnet18.yml

```

```{note}

Li et al., &ldquo;[Hermes: An Efficient Federated Learning Framework for Heterogeneous Mobile Clients](https://sites.duke.edu/angli/files/2021/10/2021_Mobicom_Hermes_v1.pdf),

&rdquo; in Proc. 27th Annual International Conference on Mobile Computing and Networking (MobiCom), 2021.

```

````
#### Personalized Federated Learning Algorithms based on Self-Supervised Learning

````{admonition} Self Supervised Learning

This category aims to achieve personalized federated learning by introducing self-supervised learning (SSL) to the training process. With SSL, an encoder model is trained to learn representations from unlabeled data. A higher performance can be achieved in subsequent tasks with the trained encoder. Only the encoder model is globally aggregated and shared during the regular training process. After reaching convergence, each client can download the trained global model to extract features from local samples. In this category, the following algorithms have been implemented:

- SimCLR [1]

- BYOL [2]

- SimSiam [3]

- MoCoV2 [4]

- SwAV [5]

- SMoG [6]

- FedEMA [7]

- Calibre

```{note}

Calibre is currently only supported on NVIDIA or M1/M2/M3 GPUs. To run on M1/M2/M3 GPUs, add the command-line argument -m.

```

```shell

uv run examples/ssl/simclr/simclr.py -c examples/ssl/configs/simclr_CIFAR10_resnet18.yml

uv run examples/ssl/byol/byol.py -c examples/ssl/configs/byol_CIFAR10_resnet18.yml

uv run examples/ssl/simsiam/simsiam.py -c examples/ssl/configs/simsiam_CIFAR10_resnet18.yml

uv run examples/ssl/moco/mocov2.py -c examples/ssl/configs/mocov2_CIFAR10_resnet18.yml

uv run examples/ssl/swav/swav.py -c examples/ssl/configs/swav_CIFAR10_resnet18.yml

uv run examples/ssl/smog/smog.py -c examples/ssl/configs/smog_CIFAR10_resnet18.yml

uv run examples/ssl/fedema/fedema.py -c examples/ssl/configs/fedema_CIFAR10_resnet18.yml

# [TODO: fix the bugs or remove it]

uv run examples/ssl/calibre/calibre.py -c examples/ssl/configs/calibre_CIFAR10_resnet18.yml

```

```{note}

[1] Chen et al., &ldquo;[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709),&rdquo; in Proc. ICML, 2020.

[2] Grill et al., &ldquo;[Bootstrap Your Own Latent A New Approach to Self-Supervised Learning](https://arxiv.org/pdf/2006.07733.pdf), &rdquo; in Proc. NeurIPS, 2020.

[3] Chen et al., &ldquo;[Exploring Simple Siamese Representation Learning](https://arxiv.org/pdf/2011.10566.pdf), &rdquo; in Proc. CVPR, 2021.

[4] Chen et al., &ldquo;[Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297), &rdquo; in ArXiv, 2020.

[5] Caron et al., &ldquo;[Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882), &rdquo; in Proc. NeurIPS, 2022.

[6] Pang et al., &ldquo;[Unsupervised Visual Representation Learning by Synchronous Momentum Grouping](https://arxiv.org/pdf/2006.07733.pdf), &rdquo; in Proc. ECCV, 2022.

[7] Zhuang et al., &ldquo;[Divergence-Aware Federated Self-Supervised Learning](https://arxiv.org/pdf/2204.04385.pdf), &rdquo; in Proc. ICLR, 2022.

```

````
