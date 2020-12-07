"""Tests for distribution generators."""

import os
import sys
import collections
import unittest
from scipy.stats import shapiro

# To import modules from the parent directory
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils import dists


class DistsTest(unittest.TestCase):
    """Tests for distribution generators."""
    def test_uniform_dist(self):
        """Test the uniform distribution generator."""
        dist, __ = dists.uniform(1000, 5)
        self.assertSequenceEqual(dist, [200] * 5)

        dist, __ = dists.uniform(101, 3)
        # The ordering of elements does not matter
        self.assertTrue(
            collections.Counter(dist) == collections.Counter([34, 34, 33]))

    def test_normal_dist(self):
        """Test the normal distribution generator."""
        dist, samples = dists.normal(100, 10)
        print('The distribution is {}'.format(dist))
        print('p-value from the Shapiro Wilk Test for normality is {0:.2f}'.
              format(shapiro(samples).pvalue))
        self.assertTrue(shapiro(samples).pvalue > 0.05)


if __name__ == '__main__':
    unittest.main()
