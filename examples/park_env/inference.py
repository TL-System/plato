from tkinter import N
from importlib_metadata import distribution
import torch
from plato.config import Config
import a2c_learning_model
import park
import csv
import os
import shutil

# Run this like
# python examples/park_env/interference.py -c examples/park_env/config_file_storage/a2c_critic_grad_lamda2_seed_15.yml


class StateNormalizer(object):
    def __init__(self, obs_space):
        self.shift = obs_space.low
        self.range = obs_space.high - obs_space.low

    def normalize(self, obs):
        return (obs - self.shift) / self.range


def evaluate_policy(model, env, obs_normalizer, dist, exp_name, eval_episodes=5):

    model.eval()

    avg_rewards = []
    for trace_idx in range(30):
        avg_reward = 0
        for _ in range(eval_episodes):
            episode_reward = 0
            done = False
            state = env.reset(trace_idx=trace_idx, test=True)
            state = obs_normalizer.normalize(state)
            steps = 0
            while not done:
                probs = model.actor(t(state))

                action = dist(probs=probs).sample()
                steps += 1
                next_state, reward, done, info = env.step(action.detach().data.numpy())
                next_state = obs_normalizer.normalize(next_state)
                state = next_state
                episode_reward += reward

            avg_reward += episode_reward
        avg_reward /= eval_episodes
        print("------------------")
        print(
            "Average Reward for model %s over trace %s for %s steps is %s"
            % (exp_name, str(trace_idx), str(steps), str(avg_reward))
        )
        print("------------------")
        avg_rewards.append(avg_reward)
    model.train()
    return avg_rewards


def t(x):
    return torch.from_numpy(x).float()


def load_model(model, filename=None, location=None):

    model_path = Config().params["model_path"] if location is None else location

    actor_model_name = "actor_model"
    critic_model_name = "critic_model"
    env_algorithm = Config().algorithm.env_name + Config().algorithm.algorithm_name
    model_seed_path = f"_seed_{Config().server.random_seed}"

    if filename is not None:
        actor_filename = f'{filename}{"_actor"}{model_seed_path}.pth'
        actor_model_path = f"{model_path}/{actor_filename}.pth"
        critic_filename = f'{filename}{"_critic"}{model_seed_path}.pth'
        critic_model_path = f"{model_path}/{critic_filename}"
    else:
        actor_model_path = (
            f"{model_path}/{env_algorithm}{actor_model_name}{model_seed_path}.pth"
        )
        critic_model_path = (
            f"{model_path}/{env_algorithm}{critic_model_name}{model_seed_path}.pth"
        )

    print("Loading actor model from %s", actor_model_path)
    print("Loading critic model from %s", critic_model_path)

    model.actor.load_state_dict(torch.load(actor_model_path), strict=True)
    model.critic.load_state_dict(torch.load(critic_model_path), strict=True)


def save(list, path):
    with open(f"{path}.csv", "w") as filehandle:
        writer = csv.writer(filehandle)
        writer.writerow(list)


if __name__ == "__main__":
    model = a2c_learning_model.Model
    model = model.get_model()
    env = park.make(Config().algorithm.env_park_name)
    normalizer = StateNormalizer(env.observation_space)
    dist = torch.distributions.Categorical

    load_model(model)
    save_list = evaluate_policy(model, env, normalizer, dist, "FedAvg")
    results_seed_path = f"{Config().results.results_dir}"
    filename = "testing_seed_" + str(Config().server.random_seed)
    path = f"{results_seed_path}/{filename}.csv"

    if not os.path.exists(results_seed_path):
        os.makedirs(results_seed_path, exist_ok=True)

    with open(path, "w") as filehandle:
        writer = csv.writer(filehandle)
        writer.writerow(save_list)
