## Examples of using Plato

The git repository includes a number of examples that showcased how third-party deep learning frameworks can be used, how customized clients and servers can be built, how Plato can be run on Google Colab, and how a collection of federated learning algorithms in the research literature can be implemented using the Plato federated learning framework.

|  Method  | Notes | Tested  |
| :------: | :---------- | :-----: |
|[Adaptive Freezing](https://henryhxu.github.io/share/chen-icdcs21.pdf) | Change directory to `examples/adaptive_freezing` and run `python adaptive_freezing.py -c <configuration file>`. | Yes |
|[Gradient-Instructed Frequency Tuning](https://github.com/TL-System/plato/blob/main/examples/adaptive_sync/papers/adaptive_sync.pdf) | Change directory to `examples/adaptive_sync` and run `python adaptive_sync.py -c <configuration file>`. | Yes |
|[Active Federated Learning](https://arxiv.org/pdf/1909.12641.pdf)| Change directory to `examples/afl` and run `python afl.py -c <configuration file>`. | Yes |
|[Attack Adaptive](https://arxiv.org/pdf/2102.05257.pdf)| Change directory to `examples/attack_adaptive` and run `python attack_adaptive.py -c <configuration file>`. | Yes |
|[Example using Catalyst](https://github.com/catalyst-team/catalyst) | This example uses a very simple model and the MNIST dataset to show how the model, the training and validation datasets, as well as the training and testing loops can be customized in Plato. It specifically uses [Catalyst](https://github.com/catalyst-team/catalyst), a popular deep learning framework, to implement the training and testing loop. Change directory to `examples/catalyst` and run `python catalyst_example.py`. | Yes |
|Customizing clients and servers | This example shows how a custom client, server, and model can be built by using class inheritance in Python. Change directory to `examples/customized` and run `python custom_server.py` to run a standalone server (with no client processes launched), then run `python custom_client.py` to start a client that connects to a server running on `localhost`. To showcase how a custom model can be used, run `python custom_model.py`. | Yes |
|Running Plato in Google Colab | This example shows how Google Colab can be used to run Plato in a terminal. Two Colab notebooks have been provided as examples, one for running Plato directly in a Colab notebook, and another for running Plato in a terminal (which is much more convenient). | Yes |
|[MistNet](https://github.com/TL-System/plato/blob/main/docs/papers/MistNet.pdf) with separate client and server implementations | Change directory to `examples/dist_mistnet` and run `python custom_server.py -c ./mistnet_lenet5_server.yml`, then run `python custom_client.py -c ./mistnet_lenet5_client.yml -i 1`. | Yes |
|[FedNova](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html) | Change directory to `examples/fednova` and run `python fednova.py -c <configuration file>`. | Yes |
|[FedSarah](https://arxiv.org/pdf/1703.00102.pdf)                             | Change directory to `examples/fedsarah` and run `python fedsarah.py -c <configuration file>`. | Yes |
|[FedProx](https://arxiv.org/pdf/1812.06127.pdf)                              | Set `optimizer` of `trainer` to `FedProx` in your configuration file. | Yes |
|[SCAFFOLD](https://arxiv.org/pdf/1910.06378.pdf)                             | Change directory to `examples/scaffold` and run `python scaffold.py -c <configuration file>`. | Not yet |
