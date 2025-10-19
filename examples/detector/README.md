# Reproducing AsyncFilter

Navigate to the examples/detector folder to start running experiments:

```bash
cd examples/detector
```

## Set up the configuration file
A variety of configuration files are provided for different experiments. Below are examples for reproducing key experiments from the paper:

### Example 1: Section 5.2 - Running AsyncFilter on CIFAR-10
```bash
uv run detector.py -c asyncfilter_cifar_2.yml
```
### Example 2: Section 5.3 - Running AsyncFilter Under LIE Attack on CINIC-10 (Concentration Factor: 0.01)
```bash
uv run detector.py -c asyncfilter_cinic_3.yml
```
### Example 3: Section 5.6 - Running AsyncFilter Under LIE Attack on FashionMNIST (Server Staleness Limit: 10)
```bash
uv run detector.py -c asyncfilter_fashionmnist_6.yml
```

Datasets download automatically on demand using the paths defined in each configuration file, so no additional flag is required.

### Customizing Experiments
For further experimentation, you can modify the configuration files to suit your requirements and reproduce the results.
