"""
The registry for samplers designed to partition the dataset across the clients.

Having a registry of all available classes is convenient for retrieving an instance based
on a configuration at run-time.
"""

import logging
from collections import OrderedDict

from plato.config import Config
from plato.samplers import (
    all_inclusive,
    dirichlet,
    distribution_noniid,
    iid,
    label_quantity_noniid,
    mixed,
    mixed_label_quantity_noniid,
    modality_iid,
    modality_quantity_noniid,
    orthogonal,
    sample_quantity_noniid,
)

registered_samplers = OrderedDict(
    [
        ("iid", iid.Sampler),
        ("noniid", dirichlet.Sampler),
        ("mixed", mixed.Sampler),
        ("orthogonal", orthogonal.Sampler),
        ("all_inclusive", all_inclusive.Sampler),
        ("distribution_noniid", distribution_noniid.Sampler),
        ("label_quantity_noniid", label_quantity_noniid.Sampler),
        ("mixed_label_quantity_noniid", mixed_label_quantity_noniid.Sampler),
        ("sample_quantity_noniid", sample_quantity_noniid.Sampler),
        ("modality_iid", modality_iid.Sampler),
        ("modality_quantity_noniid", modality_quantity_noniid.Sampler),
    ]
)


def get(datasource, client_id, testing=False, **kwargs):
    """Get an instance of the sampler."""

    sampler_type = (
        kwargs["sampler_type"]
        if "sampler_type" in kwargs
        else Config().data.testset_sampler
        if testing and hasattr(Config().data, "testset_sampler")
        else Config().data.sampler
    )
    if testing:
        logging.info("[Client #%d] Test set sampler: %s", client_id, sampler_type)
    else:
        logging.info("[Client #%d] Sampler: %s", client_id, sampler_type)

    if sampler_type in registered_samplers:
        registered_sampler = registered_samplers[sampler_type](
            datasource, client_id, testing=testing
        )
    else:
        raise ValueError(f"No such sampler: {sampler_type}")

    return registered_sampler
