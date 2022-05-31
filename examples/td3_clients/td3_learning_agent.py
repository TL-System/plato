"""
An RL agent for FL training.
"""
import logging
import math
import os
from collections import deque
from statistics import mean, stdev

import numpy as np
from plato.config import Config
from plato.utils import csv_processor
from plato.utils.reinforcement_learning import rl_agent
from plato.utils.reinforcement_learning.policies import \
    registry as policies_registry
