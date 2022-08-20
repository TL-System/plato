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

With the recent redesign of the Plato API, the following list is outdated and will be updated as they are tested again.

|  Method  | Notes | Tested  |
| :------: | :---------- | :-----: |
|[Adaptive Freezing](https://henryhxu.github.io/share/chen-icdcs21.pdf) | Change directory to `examples/adaptive_freezing` and run `python adaptive_freezing.py -c <configuration file>`. | Yes |
|[Gradient-Instructed Frequency Tuning](https://github.com/TL-System/plato/blob/main/examples/adaptive_sync/papers/adaptive_sync.pdf) | Change directory to `examples/adaptive_sync` and run `python adaptive_sync.py -c <configuration file>`. | Yes |
|[Attack Adaptive](https://arxiv.org/pdf/2102.05257.pdf)| Change directory to `examples/attack_adaptive` and run `python attack_adaptive.py -c <configuration file>`. | Yes |
|Customizing clients and servers | This example shows how a custom client, server, and model can be built by using class inheritance in Python. Change directory to `examples/customized` and run `python custom_server.py` to run a standalone server (with no client processes launched), then run `python custom_client.py` to start a client that connects to a server running on `localhost`. To showcase how a custom model can be used, run `python custom_model.py`. | Yes |
|Running Plato in Google Colab | This example shows how Google Colab can be used to run Plato in a terminal. Two Colab notebooks have been provided as examples, one for running Plato directly in a Colab notebook, and another for running Plato in a terminal (which is much more convenient). | Yes |
|[MistNet](https://github.com/TL-System/plato/blob/main/docs/papers/MistNet.pdf) with separate client and server implementations | Change directory to `examples/dist_mistnet` and run `python custom_server.py -c ./mistnet_lenet5_server.yml`, then run `python custom_client.py -c ./mistnet_lenet5_client.yml -i 1`. | Yes |
|[FedNova](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html) | Change directory to `examples/fednova` and run `python fednova.py -c <configuration file>`. | Yes |
|[FedSarah](https://arxiv.org/pdf/1703.00102.pdf)                             | Change directory to `examples/fedsarah` and run `python fedsarah.py -c <configuration file>`. | Yes |
|[SCAFFOLD](https://arxiv.org/pdf/1910.06378.pdf)                             | Change directory to `examples/scaffold` and run `python scaffold.py -c <configuration file>`. | Not yet |
