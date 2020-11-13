"""
Testing a federated learning client.
"""
import os
import sys
import unittest
import numpy as np

# To import modules from the parent directory
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from models.base import Model
from training import trainer
from servers import fedavg
from config import Config
from clients import SimpleClient

class ClientTest(unittest.TestCase):
    def setUp(self):
        __ = Config()
        self.client = SimpleClient()
        self.client.configure()
        self.client.load_data()
 
    def test_training(self):
        print("Testing training on the client...")

        report = self.client.train()
        print(report)

   
if __name__=='__main__':
    unittest.main()
