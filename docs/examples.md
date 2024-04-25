# Examples

In `examples/`, we included a wide variety of examples that showcased how third-party deep learning frameworks, such as [Catalyst](https://catalyst-team.github.io/catalyst/), can be used, and how a collection of federated learning algorithms in the research literature can be implemented using Plato by customizing the `client`, `server`, `algorithm`, and `trainer`. We also included detailed tutorials on how Plato can be run on Google Colab. Here is a list of the examples we included.

#### Support for Third-Party Frameworks 

````{admonition} **Catalyst**
Plato supports the use of third-party frameworks for its training loops. This example shows how [Catalyst](https://catalyst-team.github.io/catalyst/) can be used with Plato for local training and testing on the clients. This example uses a very simple PyTorch model and the MNIST dataset to show how the model, the training and validation datasets, as well as the training and testing loops can be quickly customized in Plato.

```shell
python examples/third_party/catalyst/catalyst_example.py -c examples/third_party/catalyst/catalyst_fedavg_lenet5.yml
```
````

#### Server Aggregation Algorithms

````{admonition} **FedAtt**
FedAtt is a server aggregation algorithm, where client updates were aggregated using a layer-wise attention-based mechanism that considered the similarity between the server and client models.  The objective was to improve the accuracy or perplexity of the trained model with the same number of communication rounds. In its implementation in `examples/fedatt/fedatt_algorithm.py`, the PyTorch implementation of FedAtt overrides `aggregate_weights()` to implement FedAtt as a custom server aggregation algorithm.

```shell
python examples/server_aggregation/fedatt/fedatt.py -c examples/server_aggregation/fedatt/fedatt_FashionMNIST_lenet5.yml
```

```{note}
S. Ji, S. Pan, G. Long, X. Li, J. Jiang, Z. Huang. &ldquo;[Learning Private Neural Language Modeling with Attentive Aggregation](https://arxiv.org/abs/1812.07108),&rdquo; in Proc. International Joint Conference on Neural Networks (IJCNN), 2019.
```
````

````{admonition} **FedAdp**
FedAdp is another server aggregation algorithm, which exploited the implicit connection between data distribution on a client and the contribution from that client to the global model, measured at the server by inferring gradient information of participating clients. In its implementation in `examples/fedadp/fedadp_server.py`, a framework-agnostic implementation of FedAdp overrides `aggregate_deltas()` to implement FedAdp as a custom server aggregation algorithm.

```shell
python examples/server_aggregation/fedadp/fedadp.py -c examples/server_aggregation/fedadp/fedadp_FashionMNIST_lenet5.yml
```

```{note}
H. Wu, P. Wang. &ldquo;[Fast-Convergent Federated Learning with Adaptive Weighting](https://ieeexplore.ieee.org/abstract/document/9442814),&rdquo; in IEEE Trans. on Cognitive Communications and Networking (TCCN), 2021.
```
````

#### Secure Aggregation with Homomorphic Encryption

````{admonition} **MaskCrypt**
MaskCrypt is a secure federated learning system based on homomorphic encryption. Instead of encrypting all the model updates, MaskCrypt encrypts only part of them to balance the tradeoff between security and efficiency. In this example, clients only select 5% of the model updates to encrypt during the learning process. The number of encrypted weights is determined by `encrypt_ratio`, which can be adjusted in the configuration file. A random mask will be adopted if `random_mask` is set to true.

```shell
python examples/secure_aggregation/maskcrypt/maskcrypt.py -c examples/secure_aggregation/maskcrypt/maskcrypt_MNIST_lenet5.yml
```
````

#### Asynchronous Federated Learning Algorithms

````{admonition} **FedAsync**
FedAsync is one of the first algorithms proposed in the literature towards operating federated learning training sessions in *asynchronous* mode, which Plato supports natively. It advocated aggregating aggressively whenever only *one* client reported its local updates to the server.

In its implementation, FedAsync's server subclasses from the `FedAvg` server and overrides its `configure()` and `aggregate_weights()` functions. In `configure()`, it needs to add some custom features (of obtaining a mixing hyperparameter for later use in the aggregation process), and calls `super().configure()` first, similar to its `__init__()` function calling `super().__init__()`. When it overrides `aggregate_weights()`, however, it supplied a completely custom implementation of this function.

```shell
python examples/async/fedasync/fedasync.py -c examples/async/fedasync/fedasync_MNIST_lenet5.yml
python examples/async/fedasync/fedasync.py -c examples/async/fedasync/fedasync_CIFAR10_resnet18.yml
```

```{note}
C. Xie, S. Koyejo, I. Gupta. &ldquo;[Asynchronous Federated Optimization](https://opt-ml.org/papers/2020/paper_28.pdf),&rdquo; in Proc. Annual Workshop on Optimization for Machine Learning (OPT), 2020.
```
````

````{admonition} **Port**
Port is one of the federated learning training sessions in *asynchronous* mode. The server will aggregate when it receives a minimum number of clients' updates, which can be tuned with 'minimum_clients_aggregated'. The 'staleness_bound' is also a common parameter in asynchronous FL, which limit the staleness of all clients' updates. 'request_update' is a special design in Port, to force clients report their updates and shut down the training process if their too slow. 'similarity_weight' and 'staleness_weight' are two hyper-parameters in Port, tuning the weights of them when the server do aggregation. 'max_sleep_time', 'sleep_simulation', 'avg_training_time' and 'simulation_distribution' are also important to define the arrival clients in Port.

```shell
python examples/async/port/port.py -c examples/async/port/port_cifar10.yml 
```

```{note}
N. Su, B. Li. &ldquo;[How Asynchronous can Federated Learning Be?](https://ieeexplore.ieee.org/document/9812885),&rdquo; in Proc. IEEE/ACM International Symposium on Quality of Service (IWQoS), 2022.
```
````

#### Federated Unlearning

````{admonition} **Federated Unlearning**
Federated unlearning is a concept proposed in the recent research literature that uses an unlearning algorithm, such as retraining from scratch, to guarantee that a client is able to remove all the effects of its local private data samples from the trained model.  In its implementation in `examples/unlearning/fedunlearning/fedunlearning_server.py` and `examples/unlearning/fedunlearning/fedunlearning_client.py`, a framework-agnostic implementation of federated unlearning overrides several methods in the client and server APIs, such as the server's `aggregate_deltas()` to implement federated unlearning.

```shell
python examples/unlearning/fedunlearning/fedunlearning.py -c examples/unlearning/fedunlearning/fedunlearning_adahessian_MNIST_lenet5.yml
```

```{note}
If the AdaHessian optimizer is used as in the example configuration file, it will reflect what the following paper proposed:

Liu et al., &ldquo;[The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid Retraining](https://arxiv.org/abs/2203.07320),&rdquo; in Proc. INFOCOM, 2022.
```
````

````{admonition} **Knot**
Knot is implemented in `examples/unlearning`, which clusters the clients, and the server aggregation is carried out within each cluster only. Knot is designed under **asynchronous** mode, and unlearned by retraining from scratch in cluster. The global model will be aggregated at the end of the retraining process. Knot supports a wide range of tasks, including image classification and language tasks.

```shell
python examples/unlearning/knot/knot.py -c examples/unlearning/knot/knot_cifar10_resnet18.yml
python examples/unlearning/knot/knot.py -c examples/unlearning/knot/knot_mnist_lenet5.yml
python examples/unlearning/knot/knot.py -c examples/unlearning/knot/knot_purchase.yml
```

```{note}
N. Su, B. Li. &ldquo;[Asynchronous Federated Unlearning](https://iqua.ece.toronto.edu/papers/ningxinsu-infocom23.pdf),&rdquo; in Proc. IEEE International Conference on Computer Communications (INFOCOM 2023).
```
````

#### Gradient Leakage Attacks and Defences

````{admonition} **Gradient leakage attacks and defenses**
Gradient leakage attacks and their defenses have been extensively studied in the research literature on federated learning.  In `examples/gradient_leakage_attacks/`, several attacks, including `DLG`, `iDLG`, and `csDLG`, have been implemented, as well as several defense mechanisms, including `Soteria`, `GradDefense`, `Differential Privacy`, `Gradient Compression`, and `Outpost`. A variety of methods in the trainer API has been used in their implementations. Refer to `examples/dlg/README.md` for more details.

```shell
python examples/gradient_leakage_attacks/dlg.py -c examples/gradient_leakage_attacks/reconstruction_emnist.yml --cpu
```
````

#### Algorithms with Customized Client Training Loops

````{admonition} **SCAFFOLD**
SCAFFOLD is a synchronous federated learning algorithm that performs server aggregation with control variates to better handle statistical heterogeneity. It has been quite widely cited and compared with in the federated learning literature. In this example, two processors, called `ExtractControlVariatesProcessor` and `SendControlVariateProcessor`, have been introduced to the client using a callback class, called `ScaffoldCallback`. They are used for sending control variates between the clients and the server. Each client also tries to maintain its own control variates for local optimization using files.

```shell
python examples/customized_client_training/scaffold/scaffold.py -c examples/customized_client_training/scaffold/scaffold_MNIST_lenet5.yml
```

```{note}
Karimireddy et al., &ldquo;[SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a.html), &rdquo; in Proc. International Conference on Machine Learning (ICML), 2020.
```
````

````{admonition} **FedProx**
To better handle system heterogeneity, the FedProx algorithm introduced a proximal term in the optimizer used by local training on the clients. It has been quite widely cited and compared with in the federated learning literature.

```shell
python examples/customized_client_training/fedprox/fedprox.py -c examples/customized_client_training/fedprox/fedprox_MNIST_lenet5.yml
```

```{note}
T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith. &ldquo;[Federated Optimization in Heterogeneous Networks](https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf),&rdquo; in Proc. Machine Learning and Systems (MLSys), 2020.
```
````

````{admonition} **FedDyn**
FedDyn is proposed to provide communication savings by dynamically updating each participating device's regularizer in each round of training. It is a method proposed to solve data heterogeneity in federated learning.
```shell
python examples/customized_client_training/feddyn/feddyn.py -c examples/customized_client_training/feddyn/feddyn_MNIST_lenet5.yml
```

```{note}
Acar, D.A.E., Zhao, Y., Navarro, R.M., Mattina, M., Whatmough, P.N. and Saligrama, V. &ldquo;[Federated learning based on dynamic regularization](https://openreview.net/forum?id=B7v4QMR6Z9w),&rdquo; Proceedings of International Conference on Learning Representations (ICLR), 2021.
```
````

````{admonition} **FedTI**
FedTI is to perform textual inversion under federated learning, which we can simply refer
to as federated textual inversion. This approach treats the learnable pseudo-word embedding as the global model, and thus allows clients to train cooperatively using the FL paradigm.

```shell
python examples/customized_client_training/fedti/fedti.py -c examples/customized_client_training/fedti/StableDiffusionFed_iid_50.yml -b FedTI
```
```{note}
Gal et al., &ldquo;[POLARIS: An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion](https://arxiv.org/pdf/2208.01618.pdf), &rdquo; Arxiv, 2022.
````

````{admonition} **FedMos**
FedMoS is a communication-efficient FL framework with coupled double momentum-based update and adaptive client selection, to jointly mitigate the intrinsic variance.

```shell
python examples/customized_client_training/fedmos/fedmos.py -c examples/customized_client_training/fedmos/fedmos_MNIST_lenet5.yml
```
```{note}
X. Wang, Y. Chen, Y. Li, X. Liao, H. Jin and B. Li, "FedMoS: Taming Client Drift in Federated Learning with Double Momentum and Adaptive Selection," IEEE INFOCOM 2023.
````


#### Client Selection Algorithms

````{admonition} **Active Federated Learning**
Active Federated Learning is a client selection algorithm, where clients were selected not uniformly at random in each round, but with a probability conditioned on the current model and the data on the client to maximize training efficiency. The objective was to reduce the number of required training iterations while maintaining the same model accuracy. In its implementation in `examples/afl/afl_server.py`, the server overrides `choose_clients()` to implement a custom client selection algorithm, and overrides `weights_aggregated()` to extract additional information from client reports.

```shell
python examples/client_selection/afl/afl.py -c examples/client_selection/afl/afl_FashionMNIST_lenet5.yml
```

```{note}
J. Goetz, K. Malik, D. Bui, S. Moon, H. Liu, A. Kumar. &ldquo;[Active Federated Learning](https://arxiv.org/abs/1909.12641),&rdquo; September 2019.
```
````

````{admonition} **Pisces**
Pisces is an asynchronous federated learning algorithm that performs biased client selection based on overall utilities and weighted server aggregation based on staleness. In this example, a client running the Pisces algorithm calculates its statistical utility and report it together with model updates to Pisces server. The server then evaluates the overall utility for each client based on the reported statistical utility and client staleness, and selects clients for the next communication round. The algorithm also attempts to detect outliers via DBSCAN for better robustness.

```shell
python examples/client_selection/pisces/pisces.py -c examples/client_selection/pisces/pisces_MNIST_lenet5.yml
```

```{note}
Jiang et al., &ldquo;[Pisces: Efficient Federated Learning via Guided Asynchronous Training](https://arxiv.org/pdf/2206.09264.pdf), &rdquo; in Proc. ACM Symposium on Cloud Computing (SoCC), 2022.
```
````

````{admonition} **Oort**
Oort is a federated learning algorithm that performs biased client selection based on both statistical utility and system utility. Originally, Oort is proposed for synchronous federated learning. In this example, it was adapted to support both synchronous and asynchronous federated learning. Notably, the Oort server maintains a blacklist for clients that have been selected too many times (10 by default). If `per_round` / `total_clients` is large, e.g. 2/5, the Oort server may not work correctly because most clients are in the blacklist and there will not be a sufficient number of clients that can be selected.

```shell
python examples/client_selection/oort/oort.py -c examples/client_selection/oort/oort_MNIST_lenet5.yml
```

```{note}
Lai et al., &ldquo;[Oort: Efficient Federated Learning via Guided Participant Selection](https://www.usenix.org/system/files/osdi21-lai.pdf),&rdquo; in Proc. USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2021.
```
````

````{admonition} **Polaris**
Polaris is a client selection method for asynchronous federated learning. In this method, it selects clients via balancing between local device speed and local data quality from an optimization perspective. As it does not require extra information rather than local updates, Polaris is pluggable to any other federated aggregation methods.

```shell
python3 examples/client_selection/polaris/polaris.py -c examples/client_selection/polaris/polaris_LeNet5.yml
```
```{note}
Kang et al., &ldquo;[POLARIS: Accelerating Asynchronous Federated Learning with Client Selection],
&rdquo; 
````

#### Split Learning Algorithms

````{admonition} **Split Learning**
Split learning aims to collaboratively train deep learning models with the server performing a portion of the training process. In split learning, each training iteration is separated into two phases: the clients first send extracted features at a specific cut layer to the server, and then the server continues the forward pass and computes gradients, which will be sent back to the clients to complete the backward pass of the training. Unlike federated learning, split learning clients sequentially interact with the server, and the global model is synchronized implicitly through the model on the server side, which is shared and updated by all clients.
```shell
python ./run -c configs/CIFAR10/split_learning_resnet18.yml
```

```{note}
Vepakomma et al., &ldquo;[Split Learning for Health: Distributed Deep Learning without Sharing Raw Patient Data](https://arxiv.org/abs/1812.00564),&rdquo; in Proc. NeurIPS, 2018.
```
````

````{admonition} **Split Learning for Training ControlNet**
ControlNet is a conditional image generation model that only finetunes the control network without updating parameters in the large diffusion model. It has a more complicated structure than the usual deep learning model. Hence, to train a ControlNet with split learning, the control network and a part of the diffusion model are on the clients and the remaining part of the diffusion model is on the server. The forwarding and backwarding processes are specifically designed according to the inputs and training targets of the image generation based on diffusion models.

```shell
python examples/split_learning/controlnet_split_learning/split_learning_main.py -c examples/split_learning/controlnet_split_learning/split_learning.yml
```
````

````{admonition} **Split Learning for Training LLM**
This is an example of fine-tuning the Hugging Face large language model with split learning. The fine-tuning policy includes training the whole model and fine-tuning with the LoRA algorithm. The cut layer in the configuration file should be set as an integer, indicating cutting at which transformer block in the transformer model.

Fine-tune the whole model
```shell
python ./examples/split_learning/llm_split_learning/split_learning_main.py -c ./examples/split_learning/llm_split_learning/split_learning_wikitext103_gpt2.yml
```
Fine-tune with LoRA
```shell
python ./examples/split_learning/llm_split_learning/split_learning_main.py -c ./examples/split_learning/llm_split_learning/split_learning_wikitext2_gpt2_lora.yml
```
````

#### Personalized Federated Learning Algorithms

````{admonition} **FedRep**
FedRep learns a shared data representation (the global layers) across clients and a unique, personalized local ``head'' (the local layers) for each client. In this implementation, after each round of local training, only the representation on each client is retrieved and uploaded to the server for aggregation.

```shell
python examples/personalized_fl/fedrep/fedrep.py -c examples/personalized_fl/configs/fedrep_CIFAR10_resnet18.yml
```

```{note}
Collins et al., &ldquo;[Exploiting Shared Representations for Personalized Federated Learning](http://proceedings.mlr.press/v139/collins21a/collins21a.pdf), &rdquo; in Proc. International Conference on Machine Learning (ICML), 2021.
```
````

````{admonition} **FedBABU**
FedBABU only updates the global layers of the model during FL training. The local layers are frozen at the beginning of each local training epoch.

```shell
python examples/personalized_fl/fedbabu/fedbabu.py -c examples/personalized_fl/configs/fedbabu_CIFAR10_resnet18.yml
```

```{note}
Oh et al., &ldquo;[FedBABU: Towards Enhanced Representation for Federated Image Classification](https://openreview.net/forum?id=HuaYQfggn5u),
&rdquo; in Proc. International Conference on Learning Representations (ICLR), 2022.
```
````

````{admonition} **APFL**
APFL jointly optimizes the global model and personalized models by interpolating between local and personalized models. Once the global model is received, each client will carry out a regular local update, and then conduct a personalized optimization to acquire a trained personalized model. The trained global model and the personalized model will subsequently be combined using the parameter "alpha," which can be dynamically updated.

```shell
python examples/personalized_fl/apfl/apfl.py -c examples/personalized_fl/configs/apfl_CIFAR10_resnet18.yml
```

```{note}
Deng et al., &ldquo;[Adaptive Personalized Federated Learning](https://arxiv.org/abs/2003.13461),
&rdquo; in Arxiv, 2021.
```
````

````{admonition} **FedPer**
FedPer learns a global representation and personalized heads, but makes simultaneous local updates for both sets of parameters, therefore makes the same number of local updates for the head and the representation on each local round.

```shell
python examples/personalized_fl/fedper/fedper.py -c examples/personalized_fl/configs/fedper_CIFAR10_resnet18.yml
```

```{note}
Arivazhagan et al., &ldquo;[Federated learning with personalization layers](https://arxiv.org/abs/1912.00818), &rdquo; in Arxiv, 2019.
```
````

````{admonition} **LG-FedAvg**
With LG-FedAvg only the global layers of a model are sent to the server for aggregation, while each client keeps local layers to itself.

```shell
python examples/personalized_fl/lgfedavg/lgfedavg.py -c examples/personalized_fl/configs/lgfedavg_CIFAR10_resnet18.yml
```

```{note}
Liang et al., &ldquo;[Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523), &rdquo; in Proc. NeurIPS, 2019.
```
````

````{admonition} **Ditto**
Ditto jointly optimizes the global model and personalized models by learning local models that are encouraged to be close together by global regularization. In this example, once the global model is received, each client will carry out a regular local update and then optimizes the personalized model.

```shell
python examples/personalized_fl/ditto/ditto.py -c examples/personalized_fl/configs/ditto_CIFAR10_resnet18.yml
```

```{note}
Li et al., &ldquo;[Ditto: Fair and robust federated learning through personalization](https://proceedings.mlr.press/v139/li21h.html), &rdquo; in Proc ICML, 2021.
```
````

````{admonition} **Per-FedAvg**
Per-FedAvg uses the Model-Agnostic Meta-Learning (MAML) framework to perform local training during the regular training rounds. It performs two forward and backward passes with fixed learning rates in each iteration.

```shell
python examples/personalized_fl/perfedavg/perfedavg.py -c examples/personalized_fl/configs/perfedavg_CIFAR10_resnet18.yml
```

```{note}
Fallah et al., &ldquo;[Personalized Federated Learning with Theoretical Guarantees: A Model-Agnostic Meta-Learning Approach](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html), &rdquo; in Proc NeurIPS, 2020.
```
````

````{admonition} **Hermes**
Hermes utilizes structured pruning to improve both communication efficiency and inference efficiency of federated learning. It prunes channels with the lowest magnitudes in each local model and adjusts the pruning amount based on each local model’s test accuracy and its previous pruning amount. When the server aggregates pruned updates, it only averages parameters that were not pruned on all clients.

```shell
python examples/personalized_fl/hermes/hermes.py -c examples/personalized_fl/configs/hermes_CIFAR10_resnet18.yml
```

```{note}
Li et al., &ldquo;[Hermes: An Efficient Federated Learning Framework for Heterogeneous Mobile Clients](https://sites.duke.edu/angli/files/2021/10/2021_Mobicom_Hermes_v1.pdf),
&rdquo; in Proc. 27th Annual International Conference on Mobile Computing and Networking (MobiCom), 2021.
```
````

#### Personalized Federated Learning Algorithms based on Self-Supervised Learning

````{admonition} **Self Supervised Learning**
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
Calibre is currently only supported on NVIDIA or M1/M2/M3 GPUs. To run on M1/M2/M3 GPUs, add the command-line argument `-m`.
```

```shell
python examples/ssl/simclr/simclr.py -c examples/ssl/configs/simclr_MNIST_lenet5.yml
python examples/ssl/simclr/simclr.py -c examples/ssl/configs/simclr_CIFAR10_resnet18.yml
python examples/ssl/byol/byol.py -c examples/ssl/configs/byol_CIFAR10_resnet18.yml
python examples/ssl/simsiam/simsiam.py -c examples/ssl/configs/simsiam_CIFAR10_resnet18.yml
python examples/ssl/moco/mocov2.py -c examples/ssl/configs/mocov2_CIFAR10_resnet18.yml
python examples/ssl/swav/swav.py -c examples/ssl/configs/swav_CIFAR10_resnet18.yml
python examples/ssl/smog/smog.py -c examples/ssl/configs/smog_CIFAR10_resnet18.yml
python examples/ssl/fedema/fedema.py -c examples/ssl/configs/fedema_CIFAR10_resnet18.yml
python examples/ssl/calibre/calibre.py -c examples/ssl/configs/calibre_CIFAR10_resnet18.yml
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

#### Algorithms based on Neural Architecture Search and Model Search

````{admonition} **FedRLNAS**
FedRLNAS is an algorithm designed to conduct Federated Neural Architecture Search without sending the entire supernet to the clients. Instead, clients still perform conventional model training as in Federated Averaging, and the server will search for the best model architecture. In this example, the server overrides ```aggregate_weights()``` to aggregate updates from subnets of different architectures into the supernet, and implements architecture parameter updates in ```weights_aggregated()```. In its implementation, only only DARTS search space is supported.

```shell
python examples/model_search/fedrlnas/fedrlnas.py -c examples/model_search/fedrlnas/FedRLNAS_MNIST_DARTS.yml
```

```{note}
Yao et al., &ldquo;[Federated Model Search via Reinforcement Learning](https://ieeexplore.ieee.org/document/9546522),&rdquo; in Proc. International Conference on Distributed Computing Systems (ICDCS), 2021.
```
````

````{admonition} **PerFedRLNAS**
PerFedRLNAS is an algorithm designed to personalize different models on each client considering data and system heterogeneity, via Federated Neural Architecture Search. Different from FedRLNAS, where the server searches a uniform architecture for all clients. In this algorithm, each client will be given a different model structure and learn personalized architecture and model weights. In this example, the update rules and sample rules are redesigned to support this feature. In current implementation, examples of NASVIT MobileNetV3, and DARTS search space are provided. 

NASVIT search space:
```shell
python ./examples/model_search/pfedrlnas/VIT/fednas.py -c ./examples/model_search/pfedrlnas/configs/PerFedRLNAS_CIFAR10_NASVIT_NonIID01.yml
```
MobileNetV3 search space (synchronous mode):
```
python ./examples/model_search/pfedrlnas/MobileNetV3/fednas.py -c ./examples/model_search/pfedrlnas/configs/PerFedRLNAS_CIFAR10_Mobilenet_NonIID03.yml
```
MobileNetV3 search space (asynchronous mode):
```
python ./examples/model_search/pfedrlnas/MobileNetV3/fednas.py -c ./examples/model_search/pfedrlnas/configs/MobileNetV3_CIFAR10_03_async.yml
```
DARTS search space
```
python ./examples/model_search/pfedrlnas/DARTS/fednas.py -c ./examples/model_search/pfedrlnas/configs/PerFedRLNAS_CIFAR10_DARTS_NonIID_03.yml
```
````

````{admonition} **FedTP**
FedTP is proposed to improve personalized federated learning with transformer structured models. For each client, the attention maps in the transformer block are generated and updated by a hypernet working on the server, instead of being updated by average aggregation. The core part is in ```fedtp_server```: ```customize_server_payload``` reloads the weights of attention maps with attention generated by hypernet before sending the models to clients and ```aggregate_weights``` updates the hypernet besides doing averaging aggregation of other parts of the model. 

```shell
python ./examples/model_search/fedtp/fedtp.py -c ./examples/model_search/fedtp/FedTP_CIFAR10_ViT_NonIID03_scratch.yml
```

```{note}
Li et al., &ldquo;[FedTP: Federated Learning by Transformer Personalization](https://arxiv.org/pdf/2211.01572v1.pdf).&rdquo; Arxiv, 2022.
````

````{admonition} **HeteroFL**
HeteroFL is an algorithm aimed at solving heterogeneous computing resources requirements on different federated learning clients. They use five different complexities to compress the channel width of the model. In the implementation, we need to modify the model to implement those five complexities and scale modules. We provide examples of ```ResNet``` family and ```MobileNetV3``` family here. The core operations of assigning different complexities to the clients and aggregate models of complexities are in function ```get_local_parameters``` and ```aggregation``` respectively, in ```heterofl_algorithm.py```.

```shell
python examples/model_search/heterofl/heterofl.py -c examples/model_search/heterofl/heterofl_resnet18_dynamic.yml
```

```{note}
Diao et al., &ldquo;[HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients](https://openreview.net/forum?id=TNkPBBYFkXg),
&rdquo; in Proc. International Conference on Learning Representations (ICLR), 2021.
````

````{admonition} **FedRolex**
FedRolex argues the statistical method of pruning channels in HeteroFL will cause unbalanced updates of the model parameters. In this algorithm, they introduce a rolling mechanism to evenly update the parameters of each channel in the system-heterogeneous federated learning. In this implementation, models of ResNet and ViT are supported.

```shell
python3 examples/model_search/fedrolex/fedrolex.py -c examples/model_search/fedrolex/example_ViT.yml
```

```{note}
Alam et al., &ldquo;[FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction](https://openreview.net/forum?id=OtxyysUdBE),
&rdquo; in Proc. NeurIPS, 2022.
```
````

````{admonition} **FjORD**
FjORD, different to FedRolex and HeteroFL, adopted a policy called ordered dropout to randomly select the pruning channels in the system-heterogeneous federated learning. To further improve the performance of aggregating models of different architectures, they further proposed to conduct distillation on each device between bigger sub-networks and smaller subnetworks, if the devices have enough computation abilities.

```shell
python3 examples/model_search/fjord/fjord.py -c examples/model_search/fjord/fjord_resnet18_dynamic.yml
```

```{note}
Samuel et al., &ldquo;[FjORD: Fair and Accurate Federated Learning under heterogeneous targets with Ordered Dropout](https://proceedings.neurips.cc/paper/2021/hash/6aed000af86a084f9cb0264161e29dd3-Abstract.html),
&rdquo; in Proc. NeurIPS, 2021.
```
````

````{admonition} **AnyCostFL**
AnyCostFL is an on-demand system-heterogeneous federated learning method to assign models of different architectures to meet the resource budgets of devices in federated learning. In this algorithm, it adopts the similar policy to assign models of different channel pruning rates as the HeteroFL. But they prune the channel on the basis of the magnitude of the $l_2$ norms of the channels. In this implementation, models of ResNet and ViT are supported.

```shell
python3 examples/model_search/anycostfl/anycostfl.py -c examples/model_search/anycostfl/example_ResNet.yml
```

```{note}
Li et al., &ldquo;[AnycostFL: Efficient On-Demand Federated Learning over Heterogeneous Edge Device](https://arxiv.org/abs/2301.03062),
&rdquo; in Proc. INFOCOM, 2022.
```
````

````{admonition} **SysHeteroFL**
In the paper system-heterogeneous federated learning revisited through architecture search, it is proposed that assigning models of different architectures to the clients to achieve better performance when there are resource budgets on the clients. In this implementation, subnets of ResNet model with different architectures are sampled.

```shell
python3 ./examples/model_search/sysheterofl/sysheterofl.py -c examples/model_search/sysheterofl/config_ResNet152.yml
```
````
#### Three-layer Federated Learning Algorithms

````{admonition} **Tempo**
Tempo is proposed to improve training performance in three-layer federated learning. It adaptively tunes the number of each client's local training epochs based on the difference between its edge server's locally aggregated model and the current global model.

```shell
python examples/three_layer_fl/tempo/tempo.py -c examples/three_layer_fl/tempo/tempo_MNIST_lenet5.yml
```

```{note}
Ying et al., &ldquo;[Tempo: Improving Training Performance in Cross-Silo Federated Learning](https://iqua.ece.toronto.edu/papers/chenying-icassp22.pdf),
&rdquo; in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.
```
````

````{admonition} **FedSaw**
FedSaw is proposed to improve training performance in three-layer federated learning with L1-norm structured pruning. Edge servers and clients pruned their updates before sending them out. FedSaw adaptively tunes the pruning amount of each edge server and its clients based on the difference between the edge server's locally aggregated model and the current global model.

```shell
python examples/three_layer_fl/fedsaw/fedsaw.py -c examples/three_layer_fl/fedsaw/fedsaw_MNIST_lenet5.yml
```
````

#### Model Pruning Algorithms

````{admonition} **FedSCR**
FedSCR uses structured pruning to prune each update’s entire filters and channels if their summed parameter values are below a particular threshold.

```shell
python examples/model_pruning/fedscr/fedscr.py -c examples/model_pruning/fedscr/fedscr_MNIST_lenet5.yml
```

```{note}
Wu et al., &ldquo;[FedSCR: Structure-Based Communication Reduction for Federated Learning](https://ieeexplore.ieee.org/document/9303442),
&rdquo; IEEE Trans. Parallel Distributed Syst., 2021.
```
````

````{admonition} **Sub-FedAvg**
Sub-FedAvg aims to obtain a personalized model for each client with non-i.i.d. local data. It iteratively prunes the parameters of each client’s local model during its local training, with the objective of removing the commonly shared parameters of local models and keeping the personalized ones. Besides the original version for two-layer federated learning, the version for three-layer federated learning has been implemented as well.

For two-layer federated learning:

```shell
python examples/model_pruning/sub_fedavg/subfedavg.py -c examples/model_pruning/sub_fedavg/subfedavg_MNIST_lenet5.yml
```

For three-layer federated learning:

```shell
python examples/model_pruning/sub_fedavg/subcs.py -c examples/model_pruning/sub_fedavg/subcs_MNIST_lenet5.yml
```

```{note}
Vahidian et al., &ldquo;[Personalized Federated Learning by Structured and Unstructured Pruning under Data Heterogeneity](https://arxiv.org/pdf/2105.00562.pdf),
&rdquo; in Proc. 41st IEEE International Conference on Distributed Computing Systems Workshops (ICDCSW), 2021.
```
````

With the redesign of the Plato API, the following list is outdated and will be updated as they are tested again.

|                                                                Method                                                                | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                     | Tested |
|:------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------:|
|                                [Adaptive Freezing](https://henryhxu.github.io/share/chen-icdcs21.pdf)                                | Change directory to `examples/adaptive_freezing` and run `python adaptive_freezing.py -c <configuration file>`.                                                                                                                                                                                                                                                                                                                           |  Yes   |
| [Gradient-Instructed Frequency Tuning](https://github.com/TL-System/plato/blob/main/examples/adaptive_sync/papers/adaptive_sync.pdf) | Change directory to `examples/adaptive_sync` and run `python adaptive_sync.py -c <configuration file>`.                                                                                                                                                                                                                                                                                                                                   |  Yes   |
|                                       [Attack Adaptive](https://arxiv.org/pdf/2102.05257.pdf)                                        | Change directory to `examples/attack_adaptive` and run `python attack_adaptive.py -c <configuration file>`.                                                                                                                                                                                                                                                                                                                               |  Yes   |
|                                                   Customizing clients and servers                                                    | This example shows how a custom client, server, and model can be built by using class inheritance in Python. Change directory to `examples/customized` and run `python custom_server.py` to run a standalone server (with no client processes launched), then run `python custom_client.py` to start a client that connects to a server running on `localhost`. To showcase how a custom model can be used, run `python custom_model.py`. |  Yes   |
|                                                    Running Plato in Google Colab                                                     | This example shows how Google Colab can be used to run Plato in a terminal. Two Colab notebooks have been provided as examples, one for running Plato directly in a Colab notebook, and another for running Plato in a terminal (which is much more convenient).                                                                                                                                                                          |  Yes   |
|   [MistNet](https://github.com/TL-System/plato/blob/main/docs/papers/MistNet.pdf) with separate client and server implementations    | Change directory to `examples/dist_mistnet` and run `python custom_server.py -c ./mistnet_lenet5_server.yml`, then run `python custom_client.py -c ./mistnet_lenet5_client.yml -i 1`.                                                                                                                                                                                                                                                     |  Yes   |
|               [FedNova](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html)               | Change directory to `examples/fednova` and run `python fednova.py -c <configuration file>`.                                                                                                                                                                                                                                                                                                                                               |  Yes   |
|                                           [FedSarah](https://arxiv.org/pdf/1703.00102.pdf)                                           | Change directory to `examples/fedsarah` and run `python fedsarah.py -c <configuration file>`.                                                                                                                                                                                                                                                                                                                                             |  Yes   |
