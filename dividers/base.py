"""
Base class for dividing data into partitions across the clients.
"""
import random


class Divider:
    """Base class for dividing data into partitions across the clients."""
    def __init__(self, dataset):
        """Get data from the dataset."""
        self.trainset = dataset.get_train_set()
        self.testset = dataset.get_test_set()
        self.labels = list(self.trainset.classes)

        random.seed()
        self.group()

    def group(self):
        """Group the training data by label."""
        # Create an empty dictionary of labels
        grouped_data = {label: [] for label in self.labels}

        # Populate the dictionary
        for datapoint in self.trainset:
            __, label = datapoint
            label = self.labels[label]

            grouped_data[label].append(datapoint)

        # Shuffling the examples in each label randomly so that
        # each client will retrieve a different partition
        for label in self.labels:
            random.shuffle(grouped_data[label])

        self.trainset = grouped_data  # Replaced with grouped data

    def extract(self, label, n):
        """Extract 'n' examples of the data for a particular label."""
        if len(self.trainset[label]) > n:
            extracted = self.trainset[label][:n]
            del self.trainset[label][:n]  # Remove from the trainset
        else:
            extracted = self.trainset[label]
            self.trainset[label] = []

        return extracted

    def trainset_size(self):
        return len(self.trainset)