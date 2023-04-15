# Configuration

This is a `hydra-core` configuration folder. There is no "full" `.yaml` file for each configuration as the full configuration is assembled from the folder structure shown here. Any parameter can be overwritten or a new parameter added at runtime using the `hydra` syntax.

**Caveat:** Overriding a whole group of options (for example when choosing a different dataset) requires the syntax `case/data=CIFAR10`!
Using only `case.data=CIFAR10` will only override the name of the dataset and does not include the full group of configurations.
