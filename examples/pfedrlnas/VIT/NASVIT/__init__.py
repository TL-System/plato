"""
The NASVIT search space is directly copyed from https://github.com/facebookresearch/NASViT.

Including models and misc, based on their codes, we did some modifications:
1. Added architect.py to support reinforcement learning space search for NAS.
2. Did some minor modifications on supernet, add its sub-classes of modules, to support quick weight copy 
between supernet and subnet.
3. You can design any search space with given NAS framework you like. For example, here we used two search space
Big and Small where Small is pruned based on origin search space provided by paper NASVIT, to save GPU (CUDA) memory cost.

Note: This can also be regarded as a routine to fit any NAS into PFedRLNAS, since PFedRLNAS algorithm
supports arbitary supernet-based Neural Architeture Search, you just do some minor modifications on source code
 of centralized NAS. Then you will be able to apply them in FedNAS with plato.

"""
