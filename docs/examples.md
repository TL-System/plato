# Examples

In `examples/`, we included a wide variety of examples that showcased how third-party deep learning frameworks, such as [Catalyst](https://catalyst-team.github.io/catalyst/), can be used, and how a collection of federated learning algorithms in the research literature can be implemented using Plato by customizing the `client`, `server`, `algorithm`, and `trainer`. We also included detailed tutorials on how Plato can be run on Google Colab. Here is a list of the examples we included.

````{admonition} **Catalyst**
Plato supports the use of third-party frameworks for its training loops. This example shows how [Catalyst](https://catalyst-team.github.io/catalyst/) can be used with Plato for local training and testing on the clients. This example uses a very simple PyTorch model and the MNIST dataset to show how the model, the training and validation datasets, as well as the training and testing loops can be quickly customized in Plato.

```shell
python examples/catalyst/catalyst_example.py -c examples/catalyst/catalyst_fedavg_lenet5.yml
```
````

````{admonition} **FedProx**
To better handle system heterogeneity, the FedProx algorithm introduced a proximal term in the optimizer used by local training on the clients. It has been quite widely cited and compared with in the federated learning literature.

```shell
python examples/fedprox/fedprox.py -c examples/fedprox/fedprox_MNIST_lenet5.yml
```

```{note}
T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith. &ldquo;[Federated Optimization in Heterogeneous Networks](https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf),&rdquo; Proceedings of Machine Learning and Systems (MLSys), 2020.
```
````

````{admonition} **FedAsync**
FedAsync is one of the first algorithms proposed in the literature towards operating federated learning training sessions in *asynchronous* mode, which Plato supports natively. It advocated aggregating aggressively whenever only *one* client reported its local updates to the server.

In its implementation, FedAsync's server subclasses from the `FedAvg` server and overrides its `configure()` and `aggregate_weights()` functions. In `configure()`, it needs to add some custom features (of obtaining a mixing hyperparameter for later use in the aggregation process), and calls `super().configure()` first, similar to its `__init__()` function calling `super().__init__()`. When it overrides `aggregate_weights()`, however, it supplied a completely custom implementation of this function.

```shell
python examples/fedasync/fedasync.py -c examples/fedasync/fedasync_MNIST_lenet5.yml
python examples/fedasync/fedasync.py -c examples/fedasync/fedasync_CIFAR10_resnet18.yml
```

```{note}
C. Xie, S. Koyejo, I. Gupta. &ldquo;[Asynchronous Federated Optimization](https://opt-ml.org/papers/2020/paper_28.pdf),&rdquo; in Proc. 12th Annual Workshop on Optimization for Machine Learning (OPT 2020).
```
````

````{admonition} **Active Federated Learning**
With Active Federated Learning as a client selection algorithm, clients were selected not uniformly at random in each round, but with a probability conditioned on the current model and the data on the client to maximize training efficiency. The objective was to reduce the number of required training iterations while maintaining the same model accuracy. In its implementation in `examples/afl/afl_server.py`, the server overrides `choose_clients()` to implement a custom client selection algorithm, and overrides `weights_aggregated()` to extract additional information from client reports.

```shell
python examples/afl/afl.py -c examples/afl/afl_FashionMNIST_lenet5.yml
```

```{note}
J. Goetz, K. Malik, D. Bui, S. Moon, H. Liu, A. Kumar. &ldquo;[Active Federated Learning](https://arxiv.org/abs/1909.12641),&rdquo; September 2019.
```
````


````{admonition} **FedAtt**
FedAtt is a server aggregation algorithm, where client updates were aggregated using a layer-wise attention-based mechanism that considered the similarity between the server and client models.  The objective was to improve the accuracy or perplexity of the trained model with the same number of communication rounds. In its implementation in `examples/fedatt/fedatt_algorithm.py`, the PyTorch implementation of FedAtt overrides `aggregate_weights()` to implement FedAtt as a custom server aggregation algorithm.

```shell
python examples/fedatt/fedatt.py -c examples/fedatt/fedatt_FashionMNIST_lenet5.yml
```

```{note}
S. Ji, S. Pan, G. Long, X. Li, J. Jiang, Z. Huang. &ldquo;[Learning Private Neural Language Modeling with Attentive Aggregation](https://arxiv.org/abs/1812.07108),&rdquo; in the Proceedings of the 2019 International Joint Conference on Neural Networks (IJCNN), March 2019.
```
````


````{admonition} **FedAdp**
FedAdp is another server aggregation algorithm, which exploited the implicit connection between data distribution on a client and the contribution from that client to the global model, measured at the server by inferring gradient information of participating clients. In its implementation in `examples/fedadp/fedadp_server.py`, a framework-agnostic implementation of FedAdp overrides `aggregate_deltas()` to implement FedAdp as a custom server aggregation algorithm.

```shell
python examples/fedadp/fedadp.py -c examples/fedadp/fedadp_FashionMNIST_lenet5.yml
```

```{note}
H. Wu, P. Wang. &ldquo;[Fast-Convergent Federated Learning with Adaptive Weighting](https://ieeexplore.ieee.org/abstract/document/9442814),&rdquo; in IEEE Transactions on Cognitive Communications and Networking (TCCN 2021).
```
````

````{admonition} **Federated Unlearning**
Federated unlearning is a concept proposed in the recent research literature that uses an unlearning algorithm, such as retraining from scratch, to guarantee that a client is able to remove all the effects of its local private data samples from the trained model.  In its implementation in `examples/fedunlearning/fedunlearning_server.py` and `examples/fedunlearning/fedunlearning_client.py`, a framework-agnostic implementation of federated unlearning overrides several methods in the client and server APIs, such as the server's `aggregate_deltas()` to implement federated unlearning.

```shell
python examples/fedunlearning/fedunlearning.py -c examples/fedunlearning/fedunlearning_adahessian_MNIST_lenet5.yml
```

```{note}
If the AdaHessian optimizer is used as in the example configuration file, it will reflect what the following paper proposed:

Liu et al., &ldquo;[The Right to be Forgotten in Federated Learning: An Efficient Realization with Rapid Retraining](https://arxiv.org/abs/2203.07320),&rdquo; in Proc. INFOCOM, 2022.
```
````

````{admonition} **Gradient leakage attacks and defenses**
Gradient leakage attacks and their defenses have been extensively studied in the research literature on federated learning.  In `examples/dlg/`, several attacks, including `DLG`, `iDLG`, and `csDLG`, have been implemented, as well as several defense mechanisms, including `Soteria`, `GradDefense`, `Differential Privacy`, `Gradient Compression`, and `Outpost`. A variety of methods in the trainer API has been used in their implementations.

```shell
python examples/dlg/dlg.py -c examples/dlg/reconstruction_emnist.yml --cpu
```
````

````{admonition} **SCAFFOLD**
SCAFFOLD is a synchronous federated learning algorithm that performs server aggregation with control variates to better handle statistical heterogeneity. It has been quite widely cited and compared with in the federated learning literature. In this example, two processors, called `ExtractControlVariatesProcessor` and `SendControlVariateProcessor`, have been introduced to the client using a callback class, called `ScaffoldCallback`. They are used for sending control variates between the clients and the server. Each client also tries to maintain its own control variates for local optimization using files.

```shell
python examples/scaffold/scaffold.py -c examples/scaffold/scaffold_MNIST_lenet5.yml
```

```{note}
Karimireddy et al., &ldquo;[SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a.html), &rdquo; in Proc. International Conference on Machine Learning (ICML), 2020.
```
````

````{admonition} **Pisces**
Pisces is an asynchronous federated learning algorithm that performs biased client selection based on overall utilities and weighted server aggregation based on staleness. In this example, a client running the Pisces algorithm calculates its statistical utility and report it together with model updates to Pisces server. The server then evaluates the overall utility for each client based on the reported statistical utility and client staleness, and selects clients for the next communication round. The algorithm also attempts to detect outliers via DBSCAN for better robustness.

```shell
python examples/pisces/pisces.py -c examples/pisces/pisces_MNIST_lenet5.yml
```

```{note}
Jiang et al., &ldquo;[Pisces: Efficient Federated Learning via Guided Asynchronous Training](https://arxiv.org/pdf/2206.09264.pdf),
&rdquo; in Proc. ACM Symposium on Cloud Computing (SoCC), 2022.
```
````

````{admonition} **Split Learning**
Split learning aims to collaboratively train deep learning models with the server performing a portion of the training process. In split learning, each training iteration is separated into two phases: the clients first send extracted features at a specific cut layer to the server, and then the server continues the forward pass and computes gradients, which will be sent back to the clients to complete the backward pass of the training. Unlike federated learning, split learning clients sequentially interact with the server, and the global model is synchronized implicitly through the model on the server side, which is shared and updated by all clients.
```shell
python examples/split_learning/split_learning.py -c examples/split_learning/split_learning_MNIST_lenet5.yml -l warn
```

```{note}
Vepakomma, et al., &ldquo;[Split Learning for Health: Distributed Deep Learning without Sharing Raw Patient Data](https://arxiv.org/abs/1812.00564),&rdquo; in Proc. AI for Social Good Workshop, affiliated with the International Conference on Learning Representations (ICLR), 2018.
```
````

````{admonition} **FedRLNAS**
FedRLNAS is an algorithm designed to conduct Federated Neural Architecture Search without sending the entire supernet to the clients. Instead, clients still perform conventional model training as in Federated Averaging, and the server will search for the best model architecture. In this example, the server overrides ```aggregate_weights()``` to aggregate updates from subnets of different architectures into the supernet, and implements architecture parameter updates in ```weights_aggregated()```. In its implementation, only only DARTS search space is supported.

```shell
python examples/fedrlnas/fedrlnas.py -c examples/fedrlnas/FedRLNAS_MNIST_DARTS.yml
```

```{note}
Yao, et al., &ldquo;[Federated Model Search via Reinforcement Learning](https://ieeexplore.ieee.org/document/9546522),&rdquo; in Proc. International Conference on Distributed Computing Systems (ICDCS), 2021.
```
````

````{admonition} **Oort**
Oort is a federated learning algorithm that performs biased client selection based on both statistical utility and system utility. Originally, Oort is proposed for synchronous federated learning. In this example, it was adapted to support both synchronous and asynchronous federated learning. Notably, the Oort server maintains a blacklist for clients that have been selected too many times (10 by defualt). If `per_round` / `total_clients` is large, e.g. 2/5, the Oort server may not work correctly because most clients are in the blacklist and there will not be a sufficient number of clients that can be selected.

```shell
python examples/oort/oort.py -c examples/oort/oort_MNIST_lenet5.yml
```

```{note}
Lai, et al., &ldquo;[Oort: Efficient Federated Learning via Guided Participant Selection](https://www.usenix.org/system/files/osdi21-lai.pdf),&rdquo; in Proc. USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2021.
```
````

````{admonition} **MaskCrypt**
MaskCrypt is a secure federated learning system based on homomorphic encryption. Instead of encrypting all the model updates, MaskCrypt encrypts only part of them to balance the tradeoff between security and efficiency. In this example, clients only select 5% of the model updates to encrypt during the learning process. The number of encrypted weights is determined by `encrypt_ratio`, which can be adjusted in the configuration file. A random mask will be adopted if `random_mask` is set to true.

```shell
python examples/maskcrypt/maskcrypt.py -c examples/maskcrypt/maskcrypt_MNIST_lenet5.yml
```
````

````{admonition} **Hermes**
Hermes utilizes structured pruning to improve both communication efficiency and inference efficiency of federated learning. It prunes channels with the lowest magnitudes in each local model and adjusts the pruning amount based on each local model’s test accuracy and its previous pruning amount. When the server aggregates pruned updates, it only averages parameters that were not pruned on all clients.

```shell
python examples/hermes/hermes.py -c examples/hermes/hermes_MNIST_lenet5.yml
```

```{note}
Li et al., &ldquo;[Hermes: An Efficient Federated Learning Framework for Heterogeneous Mobile Clients](https://sites.duke.edu/angli/files/2021/10/2021_Mobicom_Hermes_v1.pdf),
&rdquo; in Proc. 27th Annual International Conference on Mobile Computing and Networking (MobiCom), 2021.
```
````

````{admonition} **FedSCR**
FedSCR uses structured pruning to prune each update’s entire filters and channels if their summed parameter values are below a particular threshold.

```shell
python examples/fedscr/fedscr.py -c examples/fedscr/fedscr_MNIST_lenet5.yml
```

```{note}
Wu et al., &ldquo;[FedSCR: Structure-Based Communication Reduction for Federated Learning](https://ieeexplore.ieee.org/document/9303442),
&rdquo; IEEE Trans. Parallel Distributed Syst., 2021.
```
````

````{admonition} **Sub-FedAvg**
Sub-FedAvg aims to obtain a personalized model for each client with non-i.i.d. local data and reduce. It iteratively prunes the parameters of the neural networks which results in removing the commonly shared parameters of clients’ models and keeping the personalized ones. Besides the original version for two-layer federated learning, the version for three-layer federated learning has been implemented as well.

For two-layer federated learning:

```shell
python examples/sub_fedavg/subfedavg.py -c examples/sub_fedavg/subfedavg_MNIST_lenet5.yml
```

For three-layer federated learning:

```shell
python examples/sub_fedavg/subcs.py -c examples/sub_fedavg/subcs_MNIST_lenet5.yml
```

```{note}
Vahidian et al., &ldquo;[Personalized Federated Learning by Structured and Unstructured Pruning under Data Heterogeneity](https://arxiv.org/pdf/2105.00562.pdf),
&rdquo; in Proc. 41st IEEE International Conference on Distributed Computing Systems Workshops (ICDCSW), 2021.
```
````

````{admonition} **Tempo**
Tempo is proposed to improve training performance in three-layer federated learning. It adaptively tunes the number of each client's local training epochs based on the difference between its edge server's locally aggregated model and the current global model.

```shell
python examples/tempo/tempo.py -c examples/tempo/tempo_MNIST_lenet5.yml
```

```{note}
Ying et al., &ldquo;[Tempo: Improving Training Performance in Cross-Silo Federated Learning](https://iqua.ece.toronto.edu/papers/chenying-icassp22.pdf),
&rdquo; in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.
```
````

````{admonition} **FedSaw**
FedSaw is proposed to improve training performance in three-layer federated learning with L1-norm structured pruning. Edge servers and clients pruned their updates before sending them out. FedSaw adaptively tunes the pruning amount of each edge server and its clients based on the difference between the edge server's locally aggregated model and the current global model.

```shell
python examples/fedsaw/fedsaw.py -c examples/fedsaw/fedsaw_MNIST_lenet5.yml
```
````

````{admonition} **FedTP**
FedTP is proposed to improve personalized federated learning with transformer structured models. For each client, the attention maps in the transformer block are generated and updated by a hypernet working on the server, instead of being updated by average aggregation. The core part is in ```fedtp_server```: ```customize_server_payload``` reloads the weights of attention maps with attention generated by hypernet before sending the models to clients and ```aggregate_weights``` updates the hypernet besides doing averaging aggregation of other parts of the model. 

```shell
python ./examples/fedtp/fedtp.py -c ./examples/fedtp/FedTP_CIFAR10_ViT_NonIID03_scratch.yml
```

```{note}
Hongxia et al., &ldquo;[FedTP: Federated Learning by Transformer Personalization](https://arxiv.org/pdf/2211.01572v1.pdf).&rdquo;
````

````{admonition} **PerFedRLNAS**
PerFedRLNAS is an algorithm designed to personalize different models on each client considering data and system heterogeneity, via Federated Neural Architecture Search. Different from FedRLNAS, where the server searches a uniform architecture for all clients. In this algorithm, each client will be given a different model strcuture and learn personalized architecture and model weights. In this example, the update rules and sample rules are redesigned to support this feature. In current implementation, examples of NASVIT MobileNetV3, and DARTS search space are provided. 

NASVIT search space:
```shell
python ./examples/pfedrlnas/VIT/fednas.py -c ./examples/pfedrlnas/configs/PerFedRLNAS_CIFAR10_NASVIT_NonIID01.yml
```
MobileNetV3 search space (synchronous mode):
```
python ./examples/pfedrlnas/MobileNetV3/fednas.py -c ./examples/pfedrlnas/configs/PerFedRLNAS_CIFAR10_Mobilenet_NonIID03.yml
```
MobileNetV3 search space (asynchronous mode):
```
python ./examples/pfedrlnas/MobileNetV3/fednas.py -c ./examples/pfedrlnas/configs/MobileNetV3_CIFAR10_03_async.yml
```
DARTS search space
```
python ./examples/pfedrlnas/DARTS/fednas.py -c ./examples/pfedrlnas/configs/PerFedRLNAS_CIFAR10_DARTS_NonIID_03.yml
```

````

````{admonition} **FedRep**
FedRep is an algorithm for learning a shared data representation across clients and unique, personalized local ``heads'' for each client. In this implementation, after each round of local training, only the representation on each client is retrieved and uploaded to the server for aggregation.

```shell
python3 examples/fedrep/fedrep.py -c examples/fedrep/fedrep_MNIST_lenet5.yml
```

```{note}
Collins et al., &ldquo;[Exploiting Shared Representations for Personalized Federated Learning](http://proceedings.mlr.press/v139/collins21a/collins21a.pdf),
&rdquo; in International Conference on Machine Learning, 2021.
```
````

````{admonition} **FedBABU**
FedBABU argued that a better federated global model performance does not constantly improve personalization. In this algorithm, it only updates the body of the model during FL training. In this implementation, the head is frozen at the beginning of each local training epoch through the API ```train_run_start```.

```shell
python3 examples/fedbabu/fedbabu.py -c examples/fedbabu/fedavg_cifar10_levit.yml
```

```{note}
Oh et al., &ldquo;[FedBABU: Towards Enhanced Representation for Federated Image Classification](https://openreview.net/forum?id=HuaYQfggn5u),
&rdquo; in International Conference on Learning Representations (ICLR), 2022.
```
````


````{admonition} **APFL**
APFL is a synchronous personalized federated learning algorithm that jointly optimizes the global model and personalized models by interpolating between local and personalized models. It has been quite widely cited and compared with in the personalized federated learning literature. In this example, once the global model is received, each client will carry out a regular local update, and then conduct a personalized optimization to acquire a trained personalized model. The trained global model and the personalized model will subsequently be combined using the parameter "alpha," which can be dynamically updated.

```shell
python examples/apfl/apfl.py -c examples/apfl/apfl_MNIST_lenet5_noniid.yml -b NIPS
```

```{note}
Yuyang Deng, et.al., &ldquo;[Adaptive Personalized Federated Learning](https://arxiv.org/abs/2003.13461),
&rdquo; in Arxiv, 2021.
```
````

````{admonition} **FedPer**
FedPer is a synchronous personalized federated learning algorithm that learns a global representation and personalized heads, but makes simultaneous local updates for both sets of parameters, therefore makes the same number of local updates for the head and the representation on each local round.

```shell
python examples/fedper/fedper.py -c examples/fedper/fedper_MNIST_lenet5_noniid.yml -b NIPS
```

```{note}
Manoj Ghuhan Arivazhagan, et.al., &ldquo;[Federated learning with personalization layers](https://arxiv.org/abs/1912.00818),
&rdquo; in Arxiv, 2019.
````{admonition} **LG-FedAvg**
LG-FedAvg is a synchronous personalized federated learning algorithm that learns local representations and a global head. Therefore, only the head of one model is exchanged between the server and clients, while each client maintains a body of the model as its personalized encoder.

```shell
python examples/lgfedavg/lgfedavg.py -c examples/lgfedavg/lgfedavg_MNIST_lenet5_noniid.yml -b NIPS
```

```{note}
Paul Pu Liang, et.al., &ldquo;[Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523),
&rdquo; in Proc. NeurIPS, 2019.
```
````


````{admonition} **Ditto**
Ditto is another synchronous personalized federated learning algorithm that jointly optimizes the global model and personalized models by learning local models that are encouraged to be close together by global regularization. In this example, once the global model is received, each client will carry out a regular local update followed by a Ditto solver to optimize the personalized model. 

```shell
python examples/ditto/ditto.py -c examples/ditto/ditto_MNIST_lenet5_noniid.yml -b NIPS
```

```{note}
Tian Li, et.al, &ldquo;[Ditto: Fair and robust federated learning through personalization](https://proceedings.mlr.press/v139/li21h.html),
&rdquo; in Proc ICML, 2021.
```
````

````{admonition} **PerFedAvg**
PerFedAvg focuses the personalized federated learning in which our goal is to find an initial shared model that current or new users can easily adapt to their local dataset by performing one or a few steps of gradient descent with respect to their own data. Specifically, it introduces the Model-Agnostic Meta-Learning (MAML) framework into the local update of federated learning.

```shell
python examples/perfedavg/perfedavg.py -c examples/perfedavg/perfedavg_MNIST_lenet5_noniid.yml -b NIPS
```

```{note}
Alireza Fallah, et.al, &ldquo;[Ditto: Personalized federated learning with theoretical guarantees:
A model-agnostic meta-learning approach](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html),
&rdquo; in Proc NeurIPS, 2020.
```
````

````{admonition} **pFLSSL**
pFLSSL achieves Personalized federated learning by introducing self-supervised learning (SSL) to the training schema. Specifically, there are two stages. In this first stage, one global model is trained based on SSL under the federated training paradigm. Each client, in the second stage, trains its personalized model based on the extracted features of the received global model. Therefore, due to the diversity of SSL approaches, pFLSSL includes:
- SimCLR [1]
- BYOL [2]
- SimSiam [3]
- MoCoV2 [4]
- SwAV [5]
- SMoG [6]

```shell
python examples/pflSSL/simclr/simclr.py -c examples/pflSSL/simclr/simclr_MNIST_lenet5_noniid.yml -b NIPS
```

```shell
python examples/pflSSL/simclr/simclr.py -c examples/pflSSL/simclr/simclr_CIFAR10_resnet18_noniid.yml -b NIPS
```

```shell
python examples/pflSSL/byol/byol.py -c examples/pflSSL/byol/byol_MNIST_lenet5_noniid.yml -b NIPS
```

```shell
python examples/pflSSL/simsiam/simsiam.py -c examples/pflSSL/simsiam/simsiam_CIFAR10_resnet18_noniid.yml -b NIPS
```

```shell
python examples/pflSSL/moco/mocov2.py -c examples/pflSSL/moco/mocov2_CIFAR10_resnet18_noniid.yml -b NIPS
```

```shell
python examples/pflSSL/swav/swav.py -c examples/pflSSL/swav/swav_CIFAR10_resnet18_noniid.yml -b NIPS
```

```shell
python examples/pflSSL/smog/smog.py -c examples/pflSSL/smog/smog_CIFAR10_resnet18_noniid.yml -b NIPS
```

```{note}
[1]. Ting Chen, et.al., &ldquo;[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709),&rdquo; in Proc ICML, 2020.

[2]. Jean-Bastien Grill, et.al, &ldquo;[Bootstrap Your Own Latent A New Approach to Self-Supervised Learning](https://arxiv.org/pdf/2006.07733.pdf), &rdquo; in Proc NeurIPS, 2020.

[3]. Xinlei Chen, et.al, &ldquo;[Exploring Simple Siamese Representation Learning](https://arxiv.org/pdf/2011.10566.pdf), &rdquo; in ArXiv, 2020.

[4]. Xinlei Chen, et.al, &ldquo;[Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297), &rdquo; in ArXiv, 2020.

[5]. Mathilde Caron, et.al, &ldquo;[Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882), &rdquo; in Proc NeurIPS, 2020.

[6]. Bo Pang, et.al, &ldquo;[Unsupervised Visual Representation Learning by Synchronous Momentum Grouping](https://arxiv.org/pdf/2006.07733.pdf), &rdquo; in Proc ECCV, 2022.

```
````

````{admonition} **SysHeteroFL**
In the paper system-heterogneous federated learning revisited through architecture search, it is proposed that assigning models of different architectures to the clients to achieve better performance when there are resource budgets on the clients. In this implementation, subnets of ResNet model with different architectures are sampled.

```shell
python3 ./examples/sysheterofl/sysheterofl.py -c examples/sysheterofl/config_ResNet152.yml
```
````

````{admonition} **HeteroFL**
HeteroFL is an algorithm aimed at solving heterogenous computing resources requirements on different federated learning clients. They use five different complexities to compress the channel width of the model. In the implementation, we need to modify the model to implement those five complexities and scale modules. We provide examples of ```ResNet``` family and ```MobileNetV3``` family here. The core operations of assigning different complexities to the clients and aggregate models of complexities are in function ```get_local_parameters``` and ```aggregation``` respectively, in ```heterofl_algorithm.py```.

```shell
python examples/heterofl/heterofl.py -c examples/heterofl/heterofl_resnet18_dynamic.yml
```

```{note}
Diao et al., &ldquo;[HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients](https://openreview.net/forum?id=TNkPBBYFkXg),
&rdquo; in International Conference on Learning Representations (ICLR), 2021.
````

````{admonition} **FedRolex**
FedRolex argues the statistical method of pruning channels in HeteroFL will cause unbalanced updates of the model parameters. In this algorithm, they introduce a rolling mechanism to evenly update the parameters of each channel in the system-heterogenous federated learning. In this implementation, models of ResNet and ViT are supported.

```shell
python3 examples/fedrolex/fedrolex.py -c examples/fedrolex/example_ViT.yml
```

```{note}
Alam et al., &ldquo;[FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction](https://openreview.net/forum?id=OtxyysUdBE),
&rdquo; in Conference on Neural Information Processing Systems (NeurIPS), 2022.
```
````

````{admonition} **AnyCostFL**
AnyCostFL is an on-demand system-heterogeneous federated learning method to assign models of different architectures to meet the resource budgets of devices in federated learning. In this algorithm, it adopts the similar policy to assign models of different channel pruning rates as the HeteroFL. But they prune the channel on the basis of the magnitude of the $l_2$ norms of the channels. In this implementation, models of ResNet and ViT are supported.

```shell
python3 examples/anycostfl/anycostfl.py -c examples/anycostfl/example_ResNet.yml
```

```{note}
Li et al., &ldquo;[AnycostFL: Efficient On-Demand Federated Learning over Heterogeneous Edge Device](https://arxiv.org/abs/2301.03062),
&rdquo; in Proc. INFOCOM, 2022.
```
````

````{admonition} **Polaris**
Polaris is a client selection method for asynchronous federated learning. In this method, it selects clients via balancing between local device speed and local data quality from an optimization perspective. As it does not require extra information rather than local updates, Polaris is pluggable to any other federated aggregation methods.

```shell
python3 examples/polaris/polaris.py -c examples/polaris/polaris_LeNet5.yml
```
```{note}
Kang et al., &ldquo;[POLARIS: Accelerating Asynchronous Federated Learning with Client Selection],
&rdquo; 
````


With the recent redesign of the Plato API, the following list is outdated and will be updated as they are tested again.

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
