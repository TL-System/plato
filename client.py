'''
A federated learning client.
'''

import logging
import torch


class Client:
    """ A federated learning client. """

    def __init__(self, client_id):
        self.client_id = client_id


    def __repr__(self):
        return 'Client #{}: {} samples in labels: {}'.format(
            self.client_id, len(self.data), set([label for _, label in self.data]))


    # Set non-IID data configurations
    def set_bias(self, pref, bias):
        self.pref = pref
        self.bias = bias


    def set_shard(self, shard):
        self.shard = shard


    # Server interactions
    def download(self, argv):
        # Download from the server
        try:
            return argv.copy()
        except:
            return argv


    def upload(self, argv):
        # Upload to the server
        try:
            return argv.copy()
        except:
            return argv


    # Federated learning phases
    def set_data(self, data, config):
        
        # Extract from config
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

        # Extract the path from config
        model_path = self.model_path = config.general.model_path + '/' + config.general.model

        # Download from server
        config = self.download(config)

        # Extract the machine learning task from the current configuration
        self.task = config.general.task
        self.epochs = config.general.epochs
        self.batch_size = config.general.batch_size

        # Download the most recent global model
        path = model_path + '/global_model'
        self.model = model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        # Create optimizer
        self.optimizer = model.get_optimizer(self.model)


    def run(self):
        # Perform the federated learning training workload
        {
            "train": self.train()
        }[self.task]


    def get_report(self):
        # Report results to the server.
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

        # Generate report for server
        self.report = Report(self)
        self.report.weights = weights

        # Perform model testing if applicable
        if self.do_test:
            testloader = model.get_testloader(self.testset, 1000)
            self.report.accuracy = model.test(self.model, testloader)


    def test(self):
        # Perform model testing
        raise NotImplementedError


class Report(object):
    """ Federated learning client report. """

    def __init__(self, client):
        self.client_id = client.client_id
        self.num_samples = len(client.data)
