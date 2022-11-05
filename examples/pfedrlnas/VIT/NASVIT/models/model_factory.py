# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .attentive_nas_dynamic_model import AttentiveNasDynamicModel


def create_model(config, arch_config, arch=None):

    n_classes = config.parameters.model.num_classes
    bn_momentum = config.parameters.model.bn_momentum
    bn_eps = config.parameters.model.bn_eps

    # dropout = config.parameter.model.dropout
    # drop_connect = config.parameter.model.drop_connect

    model = AttentiveNasDynamicModel(
        arch_config.supernet_config,
        n_classes=n_classes,
        bn_param=(bn_momentum, bn_eps),
    )

    return model
