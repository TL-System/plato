
# Multi-modal Federated Learning
## Table of Contents
- [Introduction](#Introduction)
- [Procedure](#Procedure)
 - [References](#References)

## Introduction

Multi-modal Federated Learning involves using several sensory modalities such as memories, sound, taste and touch to train decentralized edge devices and their respective server without exchanging data among servers. We proposed Hierarchical Gradient Blending (HGB) in this site, which is an algorithm that aims to optimize th performance of Multi-modal FL in Non-iid setting. This documents mainly outlines how to set up an environment to run the codes, examples and test files.

## Procedure

### Requirements
 - youtube-dl
 - ffmpeg
 - av

You can install them using the following commands.

>
     brew install youtube-dl
     brew install ffmpeg
     pip install av


### Prepare Environment

 1. Create a conda virtual environment.
  `conda create -n PTMM2 python=3.7`

 2. Install PyTorch.
	`conda install pytorch cudatoolkit=10.1 torchvision -c pytorch`

 3. Install mmaction manually.
	 Refer to [install.md](https://github.com/open-mmlab/mmaction2/blob/master/docs/install.md) for mmaction2 and follow the steps in maual installation.
 4. Enter the following commands.
`$ pip --default-timeout 45 install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.9.0/index.html
$ mim install mmaction2`

 5. Install einops using
 `pip install einops`
 
 6. Move the *flow_extraction.py* from the **misc** directory to the **root dir of tools**
 
 7. Move the **mmaction2/tools** to the corresponding packer of the conda

 8. Move the *tests* to the corresponding packet of the conda

## References
The multi-modal environment relays on **mmaction2**. Therefore, we recommend you referece to the [mmaction site](https://github.com/open-mmlab/mmaction2) for additional information. 

[Plato](https://github.com/TL-System/plato/tree/main) is the main testing platform. 
