# Multi-Modal Federated Learning

Multi-modal federated learning involves using several sensory modalities such as memories, sound, taste and touch to train decentralized edge devices and their respective server without exchanging data among the servers. We proposed and implemented Hierarchical Gradient Blending (HGB), which is an algorithm that aims to optimize th performance of Multi-modal FL in the non-IID setting.

### Requirements

 - youtube-dl
 - ffmpeg
 - av
 - scikit-image

You can install them using the following commands:

```shell
pip install youtube-dl ffmpeg av scikit-image
```

### Preparing the Environment

 1. Create a conda virtual environment.
  `conda create -n mmplato python=3.8`

 2. Install PyTorch.
	`conda install pytorch cudatoolkit=10.1 torchvision -c pytorch`

 3. Install mmaction manually:
 
 You can install `mmcv` directly if your computer contains GPU:

 ```shell
 pip install mmcv-full
 ```
 
 If your computer only contains CPU:
 
 ```shell
 pip --default-timeout 45 install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.9.0/index.html
 ```
 
 ```shell
 git clone https://github.com/open-mmlab/mmaction2.git
 cd mmaction2
 pip install -r requirements/build.txt
 pip install -v -e .
 ```

 5. Install `einops` using
 `pip install einops`
 
 6. Prepare `mmaction2`: First move `flow_extraction.py` from the `mmaction2/tools/misc**` directory to the *root directory of tools*, and then move the `mmaction2/tools` to the corresponding `mmaction2/mmaction`.

# Non-IID Samplers

There are several types of non-IID samplers for multi-modal federated learning.

`quantity_label_noniid.py`: quantity-based label non-iid (label distribution skew), where each client contains a fixed size of classes and each class contain the almost same amount of samples.

`dirchlet.py`: distribution-based label non-iid (label distribution skew), where each client contains all classes but with different distribution. The distribution of classes within one client follows the Dirichlet distribution.

`sample_quantity_noniid.py`: quantity-based sample non-iid (quantity skew), each client contains a different amount of samples. The sample amount distribution among clients follows  the dirichlet distribution. Each client contains all classes while the number of samples in each class is almost the same.

`distribution_noniid.py`: a combination of label distribution and quantity skew.

## Additional Information

The multi-modal environment relies on `mmaction2`. You are referred to the [mmaction site](https://github.com/open-mmlab/mmaction2) for additional information.

For more detailed description on *SampleFrames* in the frames configuration files (`configs/Kinetics`) you can access:

https://github.com/open-mmlab/mmaction2/discussions/655
https://mmaction2.readthedocs.io/en/latest/tutorials/1_config.html
https://congee524-mmaction2.readthedocs.io/en/stable/config.html
