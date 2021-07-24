#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from collections import OrderedDict

from plato.config import Config

from plato.samplers import (modality_iid, quantity_modality_noniid)

registered_samplers = OrderedDict([
    ('modality_iid', modality_iid.Sampler),
    ('quantity_modality_noniid', quantity_modality_noniid.Sampler),
])


def get(datasource, client_id):
    """Get an instance of the sampler."""
    if hasattr(Config().data, 'sampler'):
        sampler_type = Config().data.modality_sampler
    else:
        sampler_type = 'modality_iid'

    logging.info("[Client #%d] Sampler: %s", client_id, sampler_type)

    if sampler_type in registered_samplers:
        registered_sampler = registered_samplers[sampler_type](datasource,
                                                               client_id)
    else:
        raise ValueError('No such sampler: {}'.format(sampler_type))

    return registered_sampler
