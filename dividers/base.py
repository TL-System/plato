"""
Base class for dividing data into partitions across the clients.
"""

class Divider:
    """Base class for dividing data into partitions across the clients."""

    def __init__(self, config, dataset):
        """Get data from the dataset."""
        self.config = config
        self.trainset = dataset.get_train_set()
        self.testset = dataset.get_test_set()
        self.labels = list(self.trainset.classes)
        self.trainset_size = len(self.trainset)

        self.group()

        # Store used data seperately
        self.used = {label: [] for label in self.labels}
        self.used['testset'] = []


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
