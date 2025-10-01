# Examples

In `examples/`, we included a wide variety of examples that showcased how third-party deep learning frameworks, such as [Catalyst](https://catalyst-team.github.io/catalyst/), can be used, and how a collection of federated learning algorithms in the research literature can be implemented using Plato by customizing the `client`, `server`, `algorithm`, and `trainer`. We also included detailed tutorials on how Plato can be run on Google Colab. Here is a list of the examples we included.

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
python examples/customized_client_training/scaffold/scaffold.py -c examples/customized_client_training/scaffold/scaffold_MNIST_lenet5.yml
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
