!!! example "dataset"
    The training and test datasets. The following options are available:

    - `HuggingFace`: including all datasets from Hugging Face (requires `dataset_name`)
    - `Torchvision`: including torchvision datasets such as MNIST, FashionMNIST, EMNIST, CIFAR10, CIFAR100, CelebA, or STL10 (requires `dataset_name`)
    - `CINIC10`
    - `FEMNIST`: Federated EMNIST
    - `TinyImageNet`
    - `Purchase`
    - `Texas`

!!! tip "Torchvision configuration"
    When using the `Torchvision` datasource, specify `dataset_name` to choose the
    dataset class exposed by `torchvision.datasets`. Optional fields include:

    - `split_parameter`: name of the constructor argument controlling the split
      (defaults to `train` or `split` when available).
    - `train_split`, `test_split`, `unlabeled_split`: values passed to the split
      parameter for each subset. For boolean splits, strings such as `"train"`
      and `"test"` map to `True` and `False`.
    - `dataset_args` / `dataset_kwargs`: positional or keyword arguments shared
      across all splits.
    - `train_args` / `train_kwargs` (and the equivalents for `test` or
      `unlabeled`): per-split overrides.
    - `download`: whether to trigger dataset downloads (defaults to `true` when
      supported by the selected dataset).
    - For EMNIST, the balanced split is assumed by default; override
      `dataset_kwargs = { split = "<variant>" }` to select a different subset.
    - For CelebA, attributes and identities are enabled by default; adjust
      `dataset_kwargs.target_type` when a different combination is required.

    !!! example "Sample Torchvision block"
        ```toml
        [data]
        datasource = "Torchvision"
        dataset_name = "MNIST"
        download = true

        dataset_kwargs = { root = "datasets" }
        train_kwargs = { train = true }
        test_kwargs = { train = false }
        ```

!!! example "data_path"
    Where the dataset is located.

    Default value: `<base_path>/data`

    !!! note "Note"
        For the `CINIC10` dataset, the default is `<base_path>/data/CINIC-10`

        For the `TinyImageNet` dataset, the default is `<base_path>/data/tiny-imagenet-200`

!!! example "train_path"
    Where the training dataset is located.

!!! example "test_path"
    Where the test dataset is located.

!!! example "sampler"
    How to divide the entire dataset to the clients. The following options are available:

    - `iid`
    - `noniid` Could have *concentration* attribute to specify the concentration parameter in the Dirichlet distribution

        !!! example "concentration"
            If the sampler is `noniid`, the concentration parameter for the Dirichlet distribution can be specified.

            Default value: `1`

    - `orthogonal` Each institution's clients have data of different classes. Could have *institution_class_ids* and *label_distribution* attributes

        !!! example "institution_class_ids"
            If the sampler is `orthogonal`, the indices of classes of local data of each institution's clients can be specified. e.g., `0, 1; 2, 3` (the first institution's clients only have data of class #0 and #1; the second institution's clients only have data of class #2 and #3).

        !!! example "label_distribution"
            If the sampler is `orthogonal`, the class distribution of every client's local data can be specified. The value should be `iid` or `noniid`.

            Default value: `iid`

    - `mixed` Some data are iid, while others are non-iid. Must have *non_iid_clients* attributes

        !!! example "non_iid_clients"
            If the sampler is `mixed`, the indices of clients whose datasets are non-i.i.d. need to be specified. Other clients' datasets are i.i.d.

!!! example "testset_sampler"
    How the test dataset is sampled when clients test locally. Any sampler type is valid.

    !!! note "Note"
        Without this parameter, the test dataset on either the client or the server is the entire test dataset of the datasource.

!!! example "random_seed"
    The random seed used to sample each client's dataset so that experiments are reproducible.

!!! example "partition_size"
    The number of samples in each client's dataset.

!!! example "testset_size"
    The number of samples in the server's test dataset when server-side evaluation is conducted.
