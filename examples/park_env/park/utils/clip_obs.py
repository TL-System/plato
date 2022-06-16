import numpy as np

from park import logger


def clip_obs(obs, obs_low, obs_high):
    # TODO: this supports 1D only, need to add a iterator in each space type
    assert len(obs.shape) == 1
    # fit in observation space
    for i in range(obs.shape[0]):
        if obs[i] > obs_high[i]:
            logger.warn('Observation at index ' + str(i) +
                ' has value ' + str(obs_arr[i]) +
                ', which is larger than obs_high ' +
                str(obs_high[i]))
            obs[i] = obs_high[i]
        if obs[i] < obs_low[i]:
            logger.warn('Observation at index ' + str(i) +
                ' has value ' + str(obs_arr[i]) +
                ', which is lower than obs_low ' +
                str(obs_low[i]))
            obs[i] = obs_low[i]
