"""
Base class for dividing data into partitions across the clients.
"""
import logging

class Divider:
    """Base class for dividing data into partitions across the clients."""

    def __init__(self, dataset):
        """Get data from the dataset."""
        self.trainset = dataset.get_train_set()
        self.testset = dataset.get_test_set()
        self.labels = list(self.trainset.classes)
        self.trainset_size = len(self.trainset)

        self.group()

        # Store used data seperately
        self.used = {label: [] for label in self.labels}
        self.used['testset'] = []


    def extract(self, label, n):
        """Extract the data for a particular label."""
        if len(self.trainset[label]) > n:
            extracted = self.trainset[label][:n]  # Extract the data
            self.used[label].extend(extracted)  # Move data to used
            del self.trainset[label][:n]  # Remove from the trainset
            return extracted
        else:
            logging.warning('Insufficient data in label: %s', label)
            logging.warning('Reusing used data.')

            # Unmark data as used
            for label in self.labels:
                self.trainset[label].extend(self.used[label])
                self.used[label] = []

            # Extract replenished data
            return self.extract(label, n)


    def group(self):
        """Group the training data by label."""
        # Create an empty dictionary of labels
        grouped_data = {label: []
                        for label in self.labels}

        # Populate the dictionary
        for datapoint in self.trainset:
            __, label = datapoint
            label = self.labels[label]

            grouped_data[label].append(datapoint)

        self.trainset = grouped_data # Replaced with grouped data
