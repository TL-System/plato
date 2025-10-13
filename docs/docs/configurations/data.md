!!! example "dataset"
    The training and test datasets. The following options are available:

    - `MNIST`
    - `FashionMNIST`
    - `EMNIST`
    - `CIFAR10`
    - `CIFAR100`
    - `CINIC10`
    - `YOLO`
    - `HuggingFace`
    - `TinyImageNet`
    - `CelebA`
    - `Purchase`
    - `Texas`
    - `STL10`

!!! example "data_path"
    Where the dataset is located.

    Default value: `./data`

    !!! note "Note"
        For the `CINIC10` dataset, the default is `./data/CINIC-10`

        For the `TinyImageNet` dataset, the default is `./data/tiny-imagenet-200`

!!! example "train_path"
    Where the training dataset is located.

    !!! note "Note"
        `train_path` need to be specified for datasets using `YOLO`.

!!! example "test_path"
    Where the test dataset is located.

    !!! note "Note"
        `test_path` need to be specified for datasets using `YOLO`.

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
