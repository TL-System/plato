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



    def extract(self, label, n):
        """Extract 'n' examples of the data for a particular label."""
        if len(self.trainset[label]) > n:
            extracted = self.trainset[label][:n]
        else:
            logging.warning('Insufficient data in label: %s', label)
            logging.warning('%s examples available, %s examples needed.', len(self.trainset[label]), n)

            extracted = self.trainset[label]

        del self.trainset[label][:n]  # Remove from the trainset
        return extracted


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
