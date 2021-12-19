import os

from plato.config import Config

os.environ['config_file'] = 'plato/utils/rlfl/fei_FMNIST_lenet5.yml'

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

        self.model_name = "td3"
        self.model_dir = os.path.join(dirname, 'models/td3')
        # self.model_dir = os.path.join(self.model_dir, 'qoe')
        self.result_dir = Config().result_dir
        self.log_interval = 10

        self.mode = 'train'  # or 'test'
        self.test_step = 100
        self.pretrained = False
        self.pretrained_iter = 0


class DDPGConfig(RLConfig):
    """Configuration specific for DDPG"""
    def __init__(self):
        super().__init__()
        # target smoothing coefficient
        self.tau = 0.005
        # self.target_update_interval = 1
        self.learning_rate = 1e-4
        # reward discounted factor
        self.gamma = 0.99
        # replay buffer size
        self.replay_size = 10000
        # mini batch size
        self.batch_size = 100
        self.sample_frequency = 2000
        self.exploration_noise = 0.1
        self.update_iteration = 1


class SACConfig(RLConfig):
    """Configuration specific for SAC"""
    def __init__(self):
        super().__init__()
        self.policy = "Gaussian"  # or "Deterministic"
        # reward discounted factor
        self.gamma = 0.99
        # target smoothing coefficient
        self.tau = 0.005
        self.learning_rate = 0.0003
        # temperature parameter determining the relative importance of
        # the entropy term against the reward
        self.alpha = 0.2
        # automaically adjust alpha
        self.automatic_entropy_tuning = False
        # random seed
        self.seed = 123456
        # mini batch size
        self.batch_size = 64
        self.hidden_size = 256
        # self.target_update_interval = 1
        self.update_iteration = 1
        # steps sampling random actions
        self.start_steps = 32
        # replay buffer size
        self.replay_size = 10000
        self.recorded_rl_items = [
            'episode', 'policy_loss', 'critic1_loss', 'critic2_loss',
            'entropy_loss', 'alpha'
        ]


class TD3Config(RLConfig):
    """Configuration specific for TD3"""
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

        self.recorded_rl_items = ['episode', 'actor_loss', 'critic_loss']

        # Whether select variable number of clients per round
        if hasattr(Config().clients, 'varied') and Config().clients.varied:
            self.varied_per_round = True
        else:
            self.varied_per_round = False
