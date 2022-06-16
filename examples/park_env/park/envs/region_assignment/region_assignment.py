import gzip
import json
import os

import numpy as np

from park import core, logger, spaces
from park.param import config
from park.utils import seeding

__FN = "park_region_assignment.json.gz"
__URL = "http://cs.brandeis.edu/~rcmarcus/park_region_assignment.json.gz"
__DATA = None


def load_data():
    """ Load region assignment data, download if needed. """
    global __DATA
    if __DATA is not None:
        return __DATA

    if not os.path.exists(__FN):
        os.system(f"wget {__URL}")

    with gzip.GzipFile(__FN, "r") as f:
        __DATA = json.load(f)
        return __DATA


class RegionAssignmentEnv(core.Env):
    """ A dynamic region assignment task.

    Assign new accounts on a social media website to regions based
    on where they are likely to be accessed from.

    * STATE *
       Estimated language of the account user
       Region the account was created in
       Whether or not the account has posted links to each of 100 sites
       
    * ACTIONS *
       Assign each account to a region (8 options)

    * REWARD *
       Total cost of serving out-of-region requests for that account

    """

    def __init__(self):
        # random seed
        self.seed(config.seed)

        self.__data = list(load_data())  # copy

        if config.ra_shuffle:
            self.np_random.shuffle(self.__data)

        self.__current_account_idx = 0

        # observation and action space
        self.__setup_space()

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def __setup_space(self):
        self.action_space = spaces.Discrete(8)

        witness = self.__data[0]

        # lang values represent a probability distribution over
        # the language of the account (or a mixture)
        self.__lang_space = spaces.Box(
            low=np.zeros(len(witness["language"])),
            high=np.ones(len(witness["language"])),
            dtype=np.float32,
        )

        # region created is one of the 8 regions where the account
        # was created
        self.__region_created_space = spaces.Discrete(8)

        # sites is some subset of a fixed set of 100 sites that
        # have been posted with the account creating the new page.
        self.__sites_space = spaces.PowerSet(set(range(100)))

        self.observation_space = spaces.Tuple(
            (self.__lang_space, self.__region_created_space, self.__sites_space)
        )

        # TODO this is correct, but may not be tight...
        self.reward_range = (-100, 100)

    def observe(self):
        curr_acct = self.__data[self.__current_account_idx]

        lang = np.array(curr_acct["language"])
        reg_created = curr_acct["region_created"]
        sites = set(np.nonzero(np.array(curr_acct["sites"]))[0])

        obs = (lang, reg_created, sites)

        assert self.__lang_space.contains(lang)
        assert self.__sites_space.contains(sites)
        assert self.__region_created_space.contains(reg_created)
        assert self.observation_space.contains(obs)
        return obs

    def step(self, action):
        assert self.action_space.contains(action)
        costs = np.array(self.__data[self.__current_account_idx]["region_costs"])

        # assume that, by placing the page in the given region, all inner-region
        # requests are free and all outside-region requests incur their cost in
        # volume
        reward = np.sum(-costs) + 2 * costs[action]

        self.__current_account_idx += 1
        done = self.__current_account_idx >= len(self.__data)
        return (self.observe() if not done else None, reward, done, {})

    def reset(self):
        self.__current_account_idx = 0
