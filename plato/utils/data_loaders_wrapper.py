"""
The implementation of various wrappers to support a flexible combination of
    dataloaders

For example, the
This is typical required in the self-supervised learning of the Plato.

For one data, it may contain:
    - trainset with labels
    - unlabeled set

The plato will create a dataloader for the trainser based on the
sampler of the trainset.

Then, the ssl-related method will also create a dataloader for the
unlabeled set based on the sampler of the unlabeled set.

Thus, there are two dataloaders to be used in the training loop for
one epoch.

"""

import numpy as np


class ParallelIter(object):
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


class CombinedBatchesLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
        taking a batch from each of them and then combining these several batches
        into one. This class mimics the `for batch in loader:` interface of
        pytorch `DataLoader`.

        Args:
            defined_loaders: a list or tuple of pytorch DataLoader objects

        The size of this dataloader equals to the minimum length of the dataloader
        within input defined loaders.
  """

    def __init__(self, defined_loaders):
        self.loaders = [
            loader for loader in defined_loaders if loader is not None
        ]

    def __iter__(self):
        return ParallelIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches


class SequentialIter(object):
    """An iterator to support iter along each loader sequentially.

        For example, there are three loaders, A, B, and, C
            the iteration will start from A,
            once A finished, B  will start
            then C will start.

        Thus, the length of this iter is the:
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


class StreamBatchesLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
        taking a batch from each of them and then combining these several batches
        into one. This class mimics the `for batch in loader:` interface of
        pytorch `DataLoader`.

        Args:
            defined_loaders: a list or tuple of pytorch DataLoader objects

        The size of this dataloader equals to the minimum length of the dataloader
        within input defined loaders.
  """

    def __init__(self, defined_loaders):
        self.loaders = [
            loader for loader in defined_loaders if loader is not None
        ]

    def __iter__(self):
        return SequentialIter(self)

    def __len__(self):
        return sum([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def process_batch(self, batch):
        return batch


# # For test:

# import torchvision
# from torch.utils.data import DataLoader
# from torchvision import transforms

# loader1 = DataLoader(torchvision.datasets.MNIST('data',
#                                                 train=True,
#                                                 download=True,
#                                                 transform=transforms.Compose([
#                                                     transforms.ToTensor(),
#                                                     transforms.Normalize(
#                                                         (0.1307, ), (0.3081, ))
#                                                 ])),
#                      batch_size=8,
#                      shuffle=True)

# loader2 = DataLoader(torchvision.datasets.MNIST('data',
#                                                 train=True,
#                                                 download=True,
#                                                 transform=transforms.Compose([
#                                                     transforms.ToTensor(),
#                                                     transforms.Normalize(
#                                                         (0.1307, ), (0.3081, ))
#                                                 ])),
#                      batch_size=64,
#                      shuffle=True)

# loader3 = DataLoader(torchvision.datasets.MNIST('data',
#                                                 train=True,
#                                                 download=True,
#                                                 transform=transforms.Compose([
#                                                     transforms.ToTensor(),
#                                                     transforms.Normalize(
#                                                         (0.1307, ), (0.3081, ))
#                                                 ])),
#                      batch_size=128,
#                      shuffle=True)

# my_parallel_loader = CombinedBatchesLoader([loader1, loader2, loader3])

# my_sequence_loader = StreamBatchesLoader([loader1, loader2, loader3, None])

# print("len(loader1): ", len(loader1))
# print("len(loader2): ", len(loader2))
# print("len(loader3): ", len(loader3))

# def loop_through_loader(loader):
#     """A use case of iterating through a mnist dataloader."""
#     for i, b1 in enumerate(loader):
#         data, target = b1
#         if i in [100, 200]:
#             print(type(data), data.size(), type(target), target.size())
#     print('num of batches: {}'.format(i + 1))

# def loop_through_my_combined_loader(loader):
#     """A use case of iterating through my_loader."""
#     stopped_batches = 0
#     for i, batches in enumerate(loader):
#         if i in [10, 20]:
#             for j, b in enumerate(batches):
#                 data, target = b
#                 print(j + 1, type(data), data.size(), type(target),
#                       target.size())
#         stopped_batches = i

#     print('stopped_batches: {}'.format(stopped_batches + 1))

# # for _ in range(4):
# #     loop_through_my_loader(my_parallel_loader)

# # print('len(my_loader):', len(my_parallel_loader))

# def loop_through_my_sequence_loader(loader):
#     """A use case of iterating through my_loader."""
#     stopped_batches = 0
#     for batch_id, batch in enumerate(loader):
#         if batch_id % 100 == 0:
#             data, target = batch
#             print(batch_id + 1, type(data), data.size(), type(target),
#                   target.size())

#         stopped_batches = batch_id

#     print('stopped_batches: {}'.format(stopped_batches + 1))

# for _ in range(4):
#     loop_through_my_sequence_loader(my_sequence_loader)

# print('len(my_loader):', len(my_sequence_loader))
