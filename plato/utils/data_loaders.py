"""
The implementation of various wrappers to support flexible
combinations of dataloaders.

Currently, two types of data loader wrappers are supported.
They are:
    - CombinedBatchesLoader
    - StreamBatchesLoader

[Example]
One specific utilization condition is self-supervised learning:
    One dataset, such as STL10, contains
    1. trainset with labels
    2. trainset without labels

The desired data loader first loads the trainset with labels and then
the one without labels.

Under this condition, the StreamBatchesLoader can be utilized to
achieve this target.

"""

import numpy as np


class ParallelIter:
    """An iterator to support iter along each loader simultaneously to generate
    one batch."""

    def __init__(self, defined_compound_loader):

        self.defined_compound_loader = defined_compound_loader
        self.compound_loaders = self.defined_compound_loader.loaders
        self.loader_iters = [iter(loader) for loader in self.compound_loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [loader_iter.next() for loader_iter in self.loader_iters]
        return self.defined_compound_loader.combine_batch(batches)

    def __len__(self):
        return len(self.defined_compound_loader)


class CombinedBatchesLoader:
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.

    :param defined_loaders: a list or tuple of pytorch DataLoader objects

    [For example]
    There are two dataloaders A and B.
    With CombinedBatchesLoader, one iter will extract one batch of samples 'A_b'
    from A and one batch of samples 'B_b' from B. Thus, the loaded term is a
    list containing [A_b, B_b].

    The size of this dataloader equals to the minimum length of the dataloader
    within input defined loaders.
    """

    def __init__(self, defined_loaders):
        self.loaders = [loader for loader in defined_loaders if loader is not None]

    def __iter__(self):
        return ParallelIter(self)

    def __len__(self):
        return min(len(loader) for loader in self.loaders)

    def combine_batch(self, batches):
        """Customize the behavior of combining batches here."""
        return batches


class SequentialIter:
    """An iterator to support iter along each loader sequentially.

    For example, there are three loaders, A, B, and, C
        the iteration will start from A,
        once A finished, B  will start
        then C will start.

    Thus, the length of this iter is:
        len(A) + len(B) + len(C)
    """

    def __init__(self, defined_compound_loader):
        # only utilize the vaild loaders

        self.defined_compound_loader = defined_compound_loader
        self.compound_loaders = self.defined_compound_loader.loaders
        self.loader_iters = [iter(loader) for loader in self.compound_loaders]

        self.loaders_len = [len(loader) for loader in self.compound_loaders]
        self.loaders_batch_bound = np.cumsum(self.loaders_len, axis=0)

        self.num_loaders = len(self.loaders_len)
        self.batch_idx = 0

    def __iter__(self):
        return self

    def __next__(self):

        # When the final loader (the last loader in the input loaders)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        cur_loader_idx = np.digitize(self.batch_idx, self.loaders_batch_bound)

        # if completed the final loader, we just recycle to the final loader
        # then, this loader will be terminated because:
        # The `StopIteration` raised inside that final loader's `__next__`
        if cur_loader_idx == self.num_loaders:
            cur_loader_idx -= 1

        loader_iter = self.loader_iters[cur_loader_idx]
        batch = loader_iter.next()

        self.batch_idx += 1

        return self.defined_compound_loader.process_batch(batch)

    def __len__(self):
        return len(self.target_loader)


class StreamBatchesLoader:
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.

    :param defined_loaders: A list or tuple containing pytorch DataLoader objects

    The size of this dataloader equals to the minimum length of the dataloader
    within input defined loaders.
    """

    def __init__(self, defined_loaders):
        self.loaders = [loader for loader in defined_loaders if loader is not None]

    def __iter__(self):
        return SequentialIter(self)

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)

    def process_batch(self, batch):
        """Customize the behavior of combining batches here."""
        return batch