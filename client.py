'''
A basic federated learning client who sends weight updates to the server.
'''

import logging
import torch

class Client:
    """ A basic federated learning client who sends simple weight updates. """

    def __init__(self, client_id):
        self.client_id = client_id
        self.do_test = None # Should the client test the trained model?
        self.test_partition = None # Percentage of the dataset reserved for testing
        self.data = None # The dataset to be used for local training
        self.trainset = None # Training dataset
        self.testset = None # Testing dataset
        self.report = None # Report to the server
        self.task = None # Local computation task: 'train' or 'test'
        self.model = None # Machine learning model
        self.epochs = None # The number of epochs in each local training round
        self.batch_size = None # The batch size used for local training


    def __repr__(self):
        return 'Client #{}: {} samples in labels: {}'.format(
            self.client_id, len(self.data), set([label for _, label in self.data]))


    def set_bias(self, pref, bias):
        self.pref = pref
        self.bias = bias


    def set_shard(self, shard):
        self.shard = shard


    def download(self, argv):
        ''' Downloading data from the server. '''
        try:
            return argv.copy()
        except:
            return argv


    def upload(self, argv):
        ''' Uploading updates to the server. '''
        try:
            return argv.copy()
        except:
            return argv


    # Federated learning phases
    def set_data(self, data, config):
        '''
        Obtaining and deploying the data from the server. For emulation purposes, all the data 
        is to be downloaded from the server.
        '''

        # Extract test parameter settings from the configuration
        do_test = self.do_test = config.clients.do_test
        test_partition = self.test_partition = config.clients.test_partition

        # Download data
        self.data = self.download(data)

        # Extract trainset, testset (if applicable)
        data = self.data
        if do_test:  # Partition for testset if applicable
            self.trainset = data[:int(len(data) * (1 - test_partition))]
            self.testset = data[int(len(data) * (1 - test_partition)):]
        else:
            self.trainset = data


    def configure(self, config):
        import model

        # Download from server
        config = self.download(config)

        # Extract the machine learning task from the current configuration
        self.task = config.general.task
        self.epochs = config.general.epochs
        self.batch_size = config.general.batch_size

        # Download the most recent global model from the server
        model_path = '{}/{}/global_model'.format(config.general.model_path, config.general.model)
        self.model = model.Net()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Create optimizer
        self.optimizer = model.get_optimizer(self.model)


    def run(self):
        # Perform the federated learning training workload
        {
            "train": self.train,
            "test": self.test
        }[self.task]()


    def get_report(self):
        ''' Report results to the server. '''
        return self.upload(self.report)


    def train(self):
        ''' The machine learning training workload on a client. '''
        import model

        logging.info('Training on client #%s', self.client_id)

        # Perform model training
        trainloader = model.get_trainloader(self.trainset, self.batch_size)
        model.train(self.model, trainloader,
                       self.optimizer, self.epochs)

        # Extract model weights and biases
        weights = model.extract_weights(self.model)

        # Generate a report for the server
        self.report = Report(self, weights)

        # Perform model testing if applicable
        if self.do_test:
            self.test()


    def test(self):
        ''' Perform model testing. '''
        import model

        testloader = model.get_testloader(self.testset, 1000)
        self.report.set_accuracy(model.test(self.model, testloader))


class Report:
    ''' Federated learning client report. '''

    def __init__(self, client, weights):
        self.client_id = client.client_id
        self.num_samples = len(client.data)
        self.weights = weights
        self.accuracy = 0

    def set_accuracy(self, accuracy):
        ''' Include the test accuracy computed at a client in the report. '''
        self.accuracy = accuracy
