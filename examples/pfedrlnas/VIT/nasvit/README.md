# Summary

This repo is the official implementation of our ICLR2022 paper ["NASVIT"](https://openreview.net/pdf?id=Qaw16njk6L). It currently includes the training/ eval code  and a pretrained supernet checkpoint on ImageNet.

# Training

`python -m torch.distributed.launch --nproc_per_node=1 --master_port=1024  main.py --cfg configs/cfg.yaml --amp-opt-level O0 --accumulation-steps 1 --batch-size 64`

# Search

Evolutionary search is done on a subsampled data set. Specifically, we randomly select five images for each category from the original ImageNet training set and treat them as our validation set.


# checkpoint 

[Download](https://drive.google.com/file/d/1Dk2yR7zHYB4dOiqCUnKjkCsKf_cMWjSY/view?usp=sharing)

**ImageNet Accuracy (val)**
| Model | Accuracy top-1 | Accuracy top-5 |
| :---: | :---: | :---: | 
| Smallest | 78.34 | 93.46 |
| Largest | 82.79 | 96.00 |

# License
The majority of NASViT is licensed under CC-BY-NC, however portions of the project are available under separate license terms: pytorch-image-models (Timm) is licensed under the Apache 2.0 license; Swin-Transformer is licensed under the MIT license.

# Contributing
We actively welcome your pull requests! Please see [CONTRIBUTING](CONTRIBUTING.md) and [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md) for more info.


