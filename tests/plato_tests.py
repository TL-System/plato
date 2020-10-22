"""Tests for distribution generators in Plato."""

import os
import sys
import collections
import unittest
import numpy as np
import random
from scipy.stats import norm, shapiro

# To import modules from the parent directory
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils import dists

class DistsTest(unittest.TestCase):
    def test_uniform_dist(self):
        dist, __ = dists.uniform(1000, 5)
        self.assertSequenceEqual(dist, [200] * 5)

        dist, __ = dists.uniform(101, 3)
        # The ordering of elements does not matter
        self.assertTrue(collections.Counter(dist) == collections.Counter([34, 34, 33]))

    def test_normal_dist(self):
        dist, samples = dists.normal(100, 10)
        print('The distribution is {}'.format(dist))
        print('p-value from the Shapiro Wilk Test for normality is {0:.2f}'.format(shapiro(samples).pvalue))
        self.assertTrue(shapiro(samples).pvalue > 0.05)


if __name__=='__main__':
      unittest.main()