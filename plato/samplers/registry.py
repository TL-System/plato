"""
The registry for samplers designed to partition the dataset across the clients.

Having a registry of all available classes is convenient for retrieving an instance based
on a configuration at run-time.
"""
import logging
from collections import OrderedDict

from plato.config import Config

if hasattr(Config().trainer, 'use_mindspore'):
    from plato.samplers.mindspore import (
        iid as iid_mindspore,
        dirichlet as dirichlet_mindspore,
    )

    registered_samplers = OrderedDict([
        ('iid', iid_mindspore.Sampler),
        ('noniid', dirichlet_mindspore.Sampler),
    ])
elif hasattr(Config().trainer, 'use_tensorflow'):
    from plato.samplers.tensorflow import base
    registered_samplers = OrderedDict([
        ('iid', base.Sampler),
        ('noniid', base.Sampler),
        ('mixed', base.Sampler),
    ])
elif hasattr(Config().trainer, 'use_nnrt'):
    # NNRT does not support dataLoader, there is no use of sampler
    from plato.samplers.nnrt import base
    registered_samplers = OrderedDict([
        ('iid', base.Sampler),
        ('noniid', base.Sampler),
        ('mixed', base.Sampler),
    ])

elif hasattr(Config.data, 'use_multimodal'):
    from plato.samplers import iid
    from plato.samplers.multimodal import (modality_iid,
                                           sample_quantity_noniid,
                                           quantity_label_noniid,
                                           quantity_modality_noniid,
                                           distribution_noniid)
    registered_samplers = OrderedDict([
        ('iid', iid.Sampler),
        ('modality_iid', modality_iid.Sampler),
        ('sample_quantity_noniid', sample_quantity_noniid.Sampler),
        ('quantity_label_noniid', quantity_label_noniid.Sampler),
        ('quantity_modality_noniid', quantity_modality_noniid.Sampler),
        ('distribution_noniid', distribution_noniid.Sampler),
    ])

else:
    from plato.samplers import (iid, dirichlet, mixed, all_inclusive)

    registered_samplers = OrderedDict([
        ('iid', iid.Sampler),
        ('noniid', dirichlet.Sampler),
        ('mixed', mixed.Sampler),
        ('all_inclusive', all_inclusive.Sampler),
    ])


def get(datasource, client_id):
    """Get an instance of the sampler."""
    if hasattr(Config().data, 'sampler'):
        sampler_type = Config().data.sampler
    else:
        sampler_type = 'iid'

    logging.info("[Client #%d] Sampler: %s", client_id, sampler_type)

    if sampler_type in registered_samplers:
        registered_sampler = registered_samplers[sampler_type](datasource,
                                                               client_id)
    else:
        raise ValueError('No such sampler: {}'.format(sampler_type))

    return registered_sampler


def multimodal_get(datasource, client_id):
    """Get an instance of the multimodal sampler."""
    if hasattr(Config().data, 'modality_sampler'):
        sampler_type = Config().data.modality_sampler
    else:
        sampler_type = 'modality_iid'

    logging.info("[Client #%d] Multimodal Sampler: %s", client_id,
                 sampler_type)

    if sampler_type in registered_samplers:
        registered_sampler = registered_samplers[sampler_type](datasource,
                                                               client_id)
    else:
        raise ValueError('No such sampler: {}'.format(sampler_type))

    return registered_sampler
