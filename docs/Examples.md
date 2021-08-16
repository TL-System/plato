## Examples of Federated Learning Algorithms

**To be completed**

Plato includes implementations of some existing federated learning algorithms, listed below.

|  Method  | Instruction | Tested  |
| :------: | :---------- | :-----: |
|[Adaptive Freezing](https://henryhxu.github.io/share/chen-icdcs21.pdf) | Change directory to `examples/adaptive_freezing` and run `python adaptive_freezing.py -c <configuration file>`. | Yes |
|[Attack Adaptive](https://arxiv.org/pdf/2102.05257.pdf)| Change directory to `examples/attack_adaptive` and run `python attack_adaptive.py -c <configuration file>`. | Yes |
|[FedNova](https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html) | Change directory to `examples/fednova` and run `python fednova.py -c <configuration file>`.| Yes |
|[FedSarah](https://arxiv.org/pdf/1703.00102.pdf)                             | Change directory to `examples/fedsarah` and run `python fedsarah.py -c <configuration file>`.| Yes |
|[FedProx](https://arxiv.org/pdf/1812.06127.pdf)                              | Set `optimizer` of `trainer` to `FedProx` in your configuration file.| Yes |
|[SCAFFOLD](https://arxiv.org/pdf/1910.06378.pdf)                             | Change directory to `examples/scaffold` and run `python scaffold.py -c <configuration file>`. | Not yet |
|[MistNet](https://github.com/TL-System/plato/blob/main/docs/papers/MistNet.pdf) with separate client and server implementations | Change directory to `examples/dist_mistnet` and run `python custom_server.py -c ./mistnet_lenet5_server.yml`, then run `python custom_client.py -c ./mistnet_lenet5_client.yml -i 1`| Yes |
