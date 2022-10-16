import random
import unittest
from collections import Counter

random.seed(0)


def getGause(scale: float, e: float = 0):
    return random.gauss(e, scale)


class ReservoirSample(object):
    def __init__(self, size):
        self._size = size
        self._counter = 0
        self._sample = []

    def feed(self, item):
        self._counter += 1
        if len(self._sample) < self._size:
            self._sample.append(item)
            return self._sample
        rand_int = random.randint(1, self._counter)
        if rand_int <= self._size:
            self._sample[rand_int - 1] = item
        return self._sample


class TestMain(unittest.TestCase):
    def test_reservoir_sample(self):
        samples = []
        for i in range(10000):
            sample = []
            rs = ReservoirSample(3)
            for item in range(1, 11):
                sample = rs.feed(item)
            samples.extend(sample)
        r = Counter(samples)
        print(r)


if __name__ == "__main__":
    unittest.main()
