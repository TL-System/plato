import os

from plato.config import Config

os.environ[
    'config_file'] = 'examples/fei/fei_FMNIST_lenet5.yml'

dirname = os.path.dirname(os.path.dirname(__file__))


class RLConfig:
    """Configuration for RL control"""
    def __init__(self):
        self.discrete_action_space = False
        self.n_actions = Config().clients.per_round
        self.per_round = Config().clients.per_round
        self.n_features = 4
        self.n_states = self.per_round * self.n_features
        self.max_action = 1
        self.min_action = -1

        self.max_episode = 2000
        self.steps_per_episode = Config().trainer.rounds
        self.target_accuracy = Config().trainer.target_accuracy
        self.device = Config().device()

        self.recorded_rl_items = ['episode', 'actor_loss', 'critic_loss']

        # Whether select variable number of clients per round
        if hasattr(Config().clients, 'varied') and Config().clients.varied:
            self.varied_per_round = True
        else:
            self.varied_per_round = False

        self.model_name = "td3"
        self.model_dir = os.path.join(dirname, 'models/td3')
        self.result_dir = Config().result_dir
        self.log_interval = 10

        self.mode = 'train'  # or 'test'
        self.test_step = 100
        self.pretrained = False
        self.pretrained_iter = 0


class TD3Config(RLConfig):
    """Configuration for TD3 policy"""
    def __init__(self):
        super().__init__()
        # reward discounted factor
        self.gamma = 0.99
        self.tau = 0.005
        self.learning_rate = 0.0003
        # random seed
        self.seed = 123456

        # Target policy smoothing is scaled wrt the action scale
        # Noise added to target policy during critic update
        self.policy_noise = 0.25 * self.max_action
        # Range to clip target policy noise
        self.noise_clip = 0.5 * self.max_action
        # Frequency of delayed policy updates
        self.policy_freq = 2

        # mini batch size
        self.batch_size = 64
        self.hidden_size = 256

        # steps sampling random actions
        self.start_steps = 32
        # replay buffer size
        self.replay_size = 10000

        # whether use LSTM or FC nets
        self.recurrent_actor = True
        self.recurrent_critic = True

