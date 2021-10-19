import os

from plato.config import Config

os.environ['config_file'] = 'plato/utils/rlfl/rl_MNIST_lenet5.yml'

dirname = os.path.dirname(os.path.dirname(__file__))

NUM_OF_STATE_FEATURES = 5


class RLConfig:
    """Configuration for RL control"""
    def __init__(self):
        self.discrete_action_space = False
        self.n_actions = Config().clients.per_round
        self.n_states = Config().clients.per_round * NUM_OF_STATE_FEATURES
        self.max_action = 1
        self.min_action = -1

        self.max_episode = 1000
        self.steps_per_episode = Config().trainer.rounds
        self.target_accuracy = Config().trainer.target_accuracy
        self.device = Config().device()

        self.model_dir = os.path.join(dirname, 'models')
        self.result_dir = Config().result_dir
        self.log_interval = 4

        self.mode = 'train'  # or 'test'
        self.test_step = 40
        self.pretrained = False
        self.pretrained_iter = 1000
