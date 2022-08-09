# Examples

In `examples/`, we included a wide variety of examples that showcased how third-party deep learning frameworks, such as [*Catalyst*](https://catalyst-team.github.io/catalyst/), can be used, and how a collection of federated learning algorithms in the research literature can be implemented using Plato by customizing the `client`, `server`, `algorithm`, and `trainer`. We also included detailed tutorials on how Plato can be run on Google Colab. Here is a list of the examples we included.

````{admonition} **Catalyst**
Plato supports the use of third-party frameworks for its training loops. This example shows how [*Catalyst*](https://catalyst-team.github.io/catalyst/) can be used with Plato for local training and testing on the clients. This example uses a very simple PyTorch model and the MNIST dataset to show how the model, the training and validation datasets, as well as the training and testing loops can be quickly customized in Plato.

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
Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). [&ldquo;Federated optimization in heterogeneous networks,&rdquo;](https://proceedings.mlsys.org/paper/2020/file/38af86134b65d0f10fe33d30dd76442e-Paper.pdf) Proceedings of Machine Learning and Systems, 2, 429-450.
```
````

With the recent redesign of the Plato API, the following list is outdated and will be updated as they are tested again.

|  Method  | Notes | Tested  |
| :------: | :---------- | :-----: |
|[Adaptive Freezing](https://henryhxu.github.io/share/chen-icdcs21.pdf) | Change directory to `examples/adaptive_freezing` and run `python adaptive_freezing.py -c <configuration file>`. | Yes |
|[Gradient-Instructed Frequency Tuning](https://github.com/TL-System/plato/blob/main/examples/adaptive_sync/papers/adaptive_sync.pdf) | Change directory to `examples/adaptive_sync` and run `python adaptive_sync.py -c <configuration file>`. | Yes |
|[Active Federated Learning](https://arxiv.org/pdf/1909.12641.pdf)| Change directory to `examples/afl` and run `python afl.py -c <configuration file>`. | Yes |
|[Attack Adaptive](https://arxiv.org/pdf/2102.05257.pdf)| Change directory to `examples/attack_adaptive` and run `python attack_adaptive.py -c <configuration file>`. | Yes |
|Customizing clients and servers | This example shows how a custom client, server, and model can be built by using class inheritance in Python. Change directory to `examples/customized` and run `python custom_server.py` to run a standalone server (with no client processes launched), then run `python custom_client.py` to start a client that connects to a server running on `localhost`. To showcase how a custom model can be used, run `python custom_model.py`. | Yes |
|Running Plato in Google Colab | This example shows how Google Colab can be used to run Plato in a terminal. Two Colab notebooks have been provided as examples, one for running Plato directly in a Colab notebook, and another for running Plato in a terminal (which is much more convenient). | Yes |
|[MistNet](https://github.com/TL-System/plato/blob/main/docs/papers/MistNet.pdf) with separate client and server implementations | Change directory to `examples/dist_mistnet` and run `python custom_server.py -c ./mistnet_lenet5_server.yml`, then run `python custom_client.py -c ./mistnet_lenet5_client.yml -i 1`. | Yes |
|[FedAdp](https://ieeexplore.ieee.org/abstract/document/9442814) | An implementation of the FedAdp algorithm — Wu et al., "[Fast-Convergent Federated Learning with Adaptive Weighting](https://ieeexplore.ieee.org/abstract/document/9442814)," in IEEE Transactions on Cognitive Communications and Networking (TCCN 2021). To run this example, change directory to `examples/fedadp`, and run `python fedadp.py -c <configuration file>`. | Yes |
|[FedAtt](https://arxiv.org/abs/1812.07108) | An implementation of the FedAtt algorithm — Ji et al., "[Learning Private Neural Language Modeling with Attentive Aggregation](https://arxiv.org/abs/1812.07108)," in the Proceedings of the 2019 International Joint Conference on Neural Networks (IJCNN 2019). To run this example, change directory to `examples/fedatt`, and run `python fedatt.py -c <configuration file>`. | Yes |
|[FedNova](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html) | Change directory to `examples/fednova` and run `python fednova.py -c <configuration file>`. | Yes |
|[FedSarah](https://arxiv.org/pdf/1703.00102.pdf)                             | Change directory to `examples/fedsarah` and run `python fedsarah.py -c <configuration file>`. | Yes |
|[SCAFFOLD](https://arxiv.org/pdf/1910.06378.pdf)                             | Change directory to `examples/scaffold` and run `python scaffold.py -c <configuration file>`. | Not yet |
