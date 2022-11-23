# Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet, [arxiv](https://arxiv.org/abs/2101.11986)

### Update:
2021/03/02: update our new results. Now T2T-ViT-14 with 21.5M parameters can reach 81.5% top1-acc by training from scratch on ImageNet. 

2021/02/21: our T2T-ViT can be trained on most of common GPUs: 1080Ti, 2080Ti, TiTAN V, V100 stably with '--amp' (Automatic Mixed Precision). In some specifical GPU like Tesla T4, 'amp' would cause NAN loss when training T2T-ViT. If you get NAN loss in training, you can disable amp by removing '--amp' in the [training scripts](https://github.com/yitu-opensource/T2T-ViT#train).


2021/02/14: update token_performer.py, now T2T-ViT-7, T2T-ViT-10, T2T-ViT-12 can be trained on 4 GPUs with 12G memory, other T2T-ViT also can be trained on 4 or 8 GPUs.

2021/01/28: init codes and upload most of the pretrained models of T2T-ViT.

<p align="center">
<img src="https://github.com/yitu-opensource/T2T-ViT/blob/main/images/f1.png">
</p>

Our codes are based on the [official imagenet example](https://github.com/pytorch/examples/tree/master/imagenet) by [PyTorch](https://pytorch.org/) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman)


## Requirements

[timm](https://github.com/rwightman/pytorch-image-models), pip install timm

torch>=1.4.0

torchvision>=0.5.0

pyyaml

data prepare: ImageNet with the following folder structure:

```
imagenet/
....train/
........calss1/
............/ img1
........class2/
............/ img2
....val/
........calss1/
............/ img1
........class2/
............/ img2
```

## T2T-ViT Models


| Model    | T2T Transformer | Top1 Acc | #params | MACs |  Download|
| :---     |   :---:         |  :---:   |  :---:  | :---: |  :---:   | 
| T2T-ViT-14   |  Performer  |   81.5   |  21.5M  | 5.2G  | [here](https://drive.google.com/file/d/19Dw1HGYkOPSwcoLZdTkmWPOMzrFMYLgN/view?usp=sharing)| 
| T2T-ViT-19   |  Performer  |   81.9   |  39.2M  | 8.9G  | [here](https://drive.google.com/file/d/1Wb476W-49TngNsXjBXChV7F2RdFv0wAA/view?usp=sharing)| 
| T2T-ViT-24   |  Performer  |   82.2   |  64.1M  | 14.1G  | [here](https://drive.google.com/file/d/1veaOABX9YmjriVYkTy2cKySiSsopxxWf/view?usp=sharing)| 
| T2T-ViT_t-14 | Transformer |   81.7   |  21.5M  | 6.1G | [here](https://drive.google.com/file/d/1WdUT-3qq3duhECKk1CabXGktvd24p3Ti/view?usp=sharing)  | 
| T2T-ViT_t-19 | Transformer |   82.4   |  39.2M  | 9.8G  | [here](https://drive.google.com/file/d/1HA15Mh7ID2XuxBilSddc5ccAggf1CD8u/view?usp=sharing) | 
| T2T-ViT_t-24 | Transformer |   82.6   |  64.1M  | 15.0G| [here](https://drive.google.com/file/d/13s9miOQt1Bo7934dyKA3m6MmPauOg5_a/view?usp=sharing) | 

The three lite variant of T2T-ViT (Comparing with MobileNets):
| Model    | T2T Transformer | Top1 Acc | #params | MACs |  Download|
| :---     |   :---:         |  :---:   |  :---:  | :---: |  :---:   | 
| T2T-ViT-7   |  Performer  |   71.7   |  4.3M   | 1.2G  | [here](https://drive.google.com/file/d/1r2Qs6MVo3fkPWwQ0hZfK4nfl6HErgdjJ/view?usp=sharing)| 
| T2T-ViT-10   |  Performer  |   75.2   |  5.9M   | 1.8G  | [here](https://drive.google.com/file/d/11v9UdXx_jw7E-lIO26MI9c29ANC2GUgw/view?usp=sharing)| 
| T2T-ViT-12   |  Performer  |   76.5   |  6.9M   | 2.2G  | [here](https://drive.google.com/file/d/1RnPvXX6HFdQ3O6B6WM1t8_SDDVBIeV6c/view?usp=sharing)  |


## Validation

Test the T2T-ViT-14 (take Performer in T2T module),

Download the [T2T-ViT-14](https://drive.google.com/file/d/1zTXtcGwIS_AmPqhUDACYDITDmnNP2yLI/view?usp=sharing), then test it by running:

```
CUDA_VISIBLE_DEVICES=0 python main.py path/to/data --model T2t_vit_14 -b 100 --eval_checkpoint path/to/checkpoint
```
The results look like:

```
Test: [   0/499]  Time: 2.083 (2.083)  Loss:  0.3578 (0.3578)  Acc@1: 96.0000 (96.0000)  Acc@5: 99.0000 (99.0000)
Test: [  50/499]  Time: 0.166 (0.202)  Loss:  0.5823 (0.6404)  Acc@1: 85.0000 (86.1569)  Acc@5: 99.0000 (97.5098)
...
Test: [ 499/499]  Time: 0.272 (0.172)  Loss:  1.3983 (0.8261)  Acc@1: 62.0000 (81.5000)  Acc@5: 93.0000 (95.6660)
Top-1 accuracy of the model is: 81.5%

```

Test the three lite variants: T2T-ViT-7, T2T-ViT-10, T2T-ViT-12 (take Performer in T2T module),

Download the [T2T-ViT-7](https://drive.google.com/file/d/1r2Qs6MVo3fkPWwQ0hZfK4nfl6HErgdjJ/view?usp=sharing), [T2T-ViT-10](https://drive.google.com/file/d/11v9UdXx_jw7E-lIO26MI9c29ANC2GUgw/view?usp=sharing) or [T2T-ViT-12](https://drive.google.com/file/d/1RnPvXX6HFdQ3O6B6WM1t8_SDDVBIeV6c/view?usp=sharing), then test it by running:

```
CUDA_VISIBLE_DEVICES=0 python main.py path/to/data --model T2t_vit_7 -b 100 --eval_checkpoint path/to/checkpoint
```


## Train

Train the three lite variants: T2T-ViT-7, T2T-ViT-10 and T2T-ViT-12 (take Performer in T2T module):

If only 4 GPUs are available,

```
CUDA_VISIBLE_DEVICES=0,1,2,3 ./distributed_train.sh 4 path/to/data --model T2t_vit_7 -b 128 --lr 1e-3 --weight-decay .03 --amp --img-size 224
```

The top1-acc in 4 GPUs would be slightly lower than 8 GPUs (around 0.1%-0.3% lower).

If 8 GPUs are available: 
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model T2t_vit_7 -b 64 --lr 1e-3 --weight-decay .03 --amp --img-size 224
```


Train the T2T-ViT-14 and T2T-ViT_t-14:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model T2t_vit_14 -b 64 --lr 5e-4 --weight-decay .05 --amp --img-size 224
```

Train the T2T-ViT-19, T2T-ViT-24 or T2T-ViT_t-19, T2T-ViT_t-24:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model T2t_vit_19 -b 64 --lr 5e-4 --weight-decay .065 --amp --img-size 224
```



## Visualization

Visualize the image features of ResNet50, you can open and run the [visualization_resnet.ipynb](https://github.com/yitu-opensource/T2T-ViT/blob/main/visualization_resnet.ipynb) file in jupyter notebook or jupyter lab; some results are given as following:

<p align="center">
<img src="https://github.com/yitu-opensource/T2T-ViT/blob/main/images/resnet_conv1.png" width="600" height="300"/>
</p>

Visualize the image features of ViT, you can open and run the [visualization_vit.ipynb](https://github.com/yitu-opensource/T2T-ViT/blob/main/visualization_vit.ipynb) file in jupyter notebook or jupyter lab; some results are given as following:

<p align="center">
<img src="https://github.com/yitu-opensource/T2T-ViT/blob/main/images/vit_block1.png" width="600" height="300"/>
</p>

Visualize attention map, you can refer to this [file](https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb). A simple example by visualizing the attention map in attention block 4 and 5 is:


<p align="center">
<img src="https://github.com/yitu-opensource/T2T-ViT/blob/main/images/attention_visualization.png" width="600" height="400"/>
</p>



## Reference
If you find this repo useful, please consider citing:
```
@article{yuan2021tokens,
  title={Tokens-to-token vit: Training vision transformers from scratch on imagenet},
  author={Yuan, Li and Chen, Yunpeng and Wang, Tao and Yu, Weihao and Shi, Yujun and Tay, Francis EH and Feng, Jiashi and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2101.11986},
  year={2021}
}
```
