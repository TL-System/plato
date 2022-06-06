"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""
import logging
from collections import OrderedDict

from plato.config import Config
from plato.utils.reinforcement_learning.policies import base, ddpg, sac, td3

registered_policies = OrderedDict([
    ('base', base.Policy),
    ('ddpg', ddpg.Policy),
    ('sac', sac.Policy),
    ('td3', td3.Policy),
])


def get(state_dim, action_space):
    """Get the DRL policy with the provided name."""
    policy_name = Config().algorithm.model_name
    logging.info("DRL Policy: %s", policy_name)

    if policy_name in registered_policies:
        registered_policy = registered_policies[policy_name](state_dim,
                                                             action_space)
    else:
        raise ValueError('No such policy: {}'.format(policy_name))

    return registered_policy
