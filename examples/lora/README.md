# Fine-tuning Large Language Models using Federated Learning with LoRA Adapter

Implementation of LoRA fine-tuning for LLMs using the Plato framework.

---

## Run the Example

1. Make sure the `pip` packages required for this algorithm are installed before running the program:
```
pip install -r requirements.txt --upgrade
```
and the current GitHub version of Plato as a local `pip` package is installed:
```
pip install .
```

2. a. To run the example server and clients on the same machine, use command
```
python examples/lora/lora.py -c examples/lora/fedavg_opt.yml
```

2. b. Or, to run the example on different machines, start the server first by command
```
python examples/lora/lora_server.py -c examples/lora/server.yml
```
and then start clients by command
```
python examples/lora/lora_client.py -c examples/lora/client.yml -i <client_id>
```
where the server's address should be specified in `server:address` in `client.yml`.


## Configurations

To experiment with a different HuggingFace dataset or model, change `data:dataset_name` and `trainer:model_name` in the `.yml` file. To tune LoRA related hyper-parameters, modify the elements under the section `parameters:lora` in the `.yml` file.
