# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .attentive_nas_dynamic_model import AttentiveNasDynamicModel


def create_model(args, arch=None):
    n_classes = int(getattr(args, "n_classes", 1000))
    bn_momentum = getattr(args, "bn_momentum", 0.1)
    bn_eps = getattr(args, "bn_eps", 1e-5)

    dropout = getattr(args, "dropout", 0)
    drop_connect = getattr(args, "drop_connect", 0)

    if arch is None:
        arch = args.arch

    if arch == "attentive_nas_dynamic_model":
        model = AttentiveNasDynamicModel(
            args.supernet_config,
            n_classes=n_classes,
            bn_param=(bn_momentum, bn_eps),
        )
    elif arch == "attentive_nas_static_model":
        supernet = AttentiveNasDynamicModel(
            args.supernet_config,
            n_classes=n_classes,
            bn_param=(bn_momentum, bn_eps),
        )
        # load from pretrained models
        supernet.load_weights_from_pretrained_models(
            args.pareto_models.supernet_checkpoint_path
        )

        # subsample a static model with weights inherited from the supernet dynamic model
        supernet.set_active_subnet(
            resolution=args.active_subnet.resolution,
            width=args.active_subnet.width,
            depth=args.active_subnet.depth,
            kernel_size=args.active_subnet.kernel_size,
            expand_ratio=args.active_subnet.expand_ratio,
        )
        model = supernet.get_active_subnet()

        # house-keeping stuff
        model.set_bn_param(momentum=bn_momentum, eps=bn_eps)
        del supernet
    else:
        raise ValueError(arch)

    return model
