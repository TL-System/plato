# Code Structure and Implementation methods


We first implementate the basic framework for the personalized federated learning (pFL). The corresponding code is placed under:

> Folder structure and functions for pFL

    ├── trainers
    ├── ── pers_basic.py                    # Basic Trainer for pFL
    ├── ── contrastive_ssl.py               # pFL-SSL's trainer built upon pers_basic.py
    ├── servers
    ├── ── fedavg_pers.py                   # Basic Server for pFL
    ├── algorithms
    ├── ── fedavg_pers.py                   # Basic Algorithm for pFL
    ├── clients
    ├── ── pers_simple.py                   # Basic Clients for pFL
    ├── ── ssl_simple.py                    # pFL-SSL's clients built upon pers_simple.py


Then, based on the pFL's implementation, we build the pFL-SSL framework for Plato.
> Folder structure and functions for pFL-SSL

    ├── datasources
    ├── ── augmentations/*                  # All data augmentations used for pFL-SSL
    ├── ── stl10.py                         # The STL10 dataset used in the experiments
    ├── ── contrastive_data_wrapper.py      # The Data wrapper to support the contrastive samples from basic Plato's datasources
    ├── ── datawrapper_registry.py          # The register for the data wrapper
    ├── models
    ├── ── encoders_register.py             # Generate Encoder from basic networks such as LeNet5, ResNet*
    ├── ── general_mlps_register.py         # Generate any MLPs networks
    ├── ── ssl_monitor_register.py          # Generate the monitor (KNN) for pFL-SSL
    ├── samplers                            # We slightly modified the sampler to support the unlabelled dataset
    ├── trainers
    ├── ── contrastive_ssl.py               # pFL-SSL's trainer built upon pers_basic.py
    ├── clients
    ├── ── ssl_simple.py                    # pFL-SSL's clients built upon pers_simple.py
    ├── utils
    ├── ── arrange_saving_name.py           # Generate consistent file name for all saving operation used in pFL-SSL
    ├── ── checkpoint_operator.py           # Saving and Loading operations, also support load latest checkpoint file for clients
    ├── ── csv_processor.py                 # Write and Read the csv files to record the results.
    ├── ── data_loaders_wrapper.py          # To support a stream data loaders that combines two different data loaders, used in STL10 dataset
    ├── ── lars_optimizer.py                # The LARs optimizer used to train SSL methods.
    ├── ── optimizers.py                    # Added code to support the behavior of defining optimizer and lr schedule for general training
                                            # and personalized training process, i.e., pre-training and personalization stage in the paper.
    ├── config.py                           # Modified the config.py to support many required features in our paper
                                            # such as
                                            #   - preparing the experiment saving dir based on the run method and dataset,...
                                            #   - Saving the running config file (xxx.yml) to the corresponding results dir


The pFL-SSL framework is built upon the self-supervised learning. For the details framework design, please access the [PR#200](https://github.com/TL-System/plato/pull/200). We did not make major modifications to this original design.

Then, we build our proposed framework pFL-SSL upon [BYOL](https://papers.nips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf), [SimCLR](https://arxiv.org/abs/2002.05709) , [MoCoV2](https://arxiv.org/abs/2003.04297) , and [Simsiam](https://arxiv.org/abs/2011.10566) to generate benchmark methods used in our paper, pFL-BYOL, pFL-SimCLR, pFL-MoCov2and pFL-Simsiam, respectively.

Then, for pFL methods relying meta-learning methods, we implemented [P-FedAvg](https://arxiv.org/abs/1909.12488?context=cs#:~:text=Improving%20Federated%20Learning%20Personalization%20via%20Model%20Agnostic%20Meta%20Learning,-Yihan%20Jiang%2C%20Jakub&text=Federated%20Learning%20(FL)%20refers%20to,the%20activity%20of%20their%20users.) and [Per-FedAvg](https://proceedings.neurips.cc/paper/2020/hash/24389bfe4fe2eba8bf9aa9203a44cdad-Abstract.html).

Finally, we implemented one federated representation learning method for pFL, [FedRep](http://proceedings.mlr.press/v139/collins21a/collins21a.pdf), based on the uppermentioned pFL architecture.

Then, source code for the implemented methods is stored under:

> Folder structure and functions for implemented methods

    ├── byol/*                      # pFL-BYOL method
    ├── simclr/*                    # pFL-SimCLR method
    ├── mocov2/*                    # pFL-MoCov2and method
    ├── simsiam/*                   # pFL-Simsiam method
    ├── fedavg/*                    # P-FedAvg method, we simplify the name.
    ├── perfedavg/*                 # Per-FedAvg method
    ├── fedrep/*                    # FedRep method
    ├── pFLCMA/*                    # our proposed pFLCMA method


Finally, we prepare some useful tools to run the code:

> Functions for code running

    ├── generate_run_scripts.py                 # Generate run script (.sh) used in Sim server for all configs/*
    ├── generate_unique_port_id.py              # Generate unique 'port id' for config files under configs/* to
    ├── submit_slurm_jobs.py                    # Submit Slurm jobs based on the requirements
    ├── extract_outputs_to_local.py             # Extract obtained experimental results from Sim to local

All config files are stored under the `configs/` dir.

# Some examples to run the code

```SimCLR
python examples/contrastive_adaptation/simclr/simclr.py -c examples/contrastive_adaptation/configs/whole_global_model/simclr_CIFAR10_resnet18.yml  -b ./INFOCOM23/experiments/whole_global_model
```

```BYOL
python examples/contrastive_adaptation/byol/byol.py -c examples/contrastive_adaptation/configs/whole_global_model/byol_CIFAR10_resnet18.yml -b ./INFOCOM23/experiments/whole_global_model
```

```SimSiam
python examples/contrastive_adaptation/simsiam/simsiam.py -c examples/contrastive_adaptation/configs/whole_global_model/simsiam_CIFAR10_resnet18.yml -b ./INFOCOM23/experiments/whole_global_model
```

```MoCo
python examples/contrastive_adaptation/moco/moco.py -c examples/contrastive_adaptation/configs/whole_global_model/moco_CIFAR10_resnet18.yml -b ./INFOCOM23/experiments/whole_global_model/whole_global_model
```

The more general way is to run `generate_run_scripts.py` and then `generate_unique_port_id.py` in the Sim server to generate the job running scripts. Then, the user can submit the corresponding jobs by using `submit_slurm_jobs.py` with key words or directly submit the script files under `run_scripts/*`.



