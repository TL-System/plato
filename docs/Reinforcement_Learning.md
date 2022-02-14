# Reinforcement_Learning: A Reinforcement Learning Framework for Active Federated Learning

_The programs of the framework is located at `plato/utils/reinforcement_learning/`._

## Description to primary files

- `simple_rl_agent.py`: A simple RL agent that uses the Gym interface to realize the RL control.
- `simple_rl_server.py`: A simple FL server inherited from `plato.servers.base` that conducts the FL training with the control of an RL agent.
- `policies`: Under this directory, several RL control policies and NNs have been implemented for experiments.

## How to use this framework and customize your own DRL agent for learning and controlling FL?

There is an example project in `examples/fei` that has implemented an DRL agent to learn and control the global aggregation step in FL.

Just like it, we can customize another DRL agent by:

1. Create a new directory `examples\your_algorithm` and then fulfill the following files in this directory

2. Implement a subclass of `RLAgent` in `simple_rl_agent.py` and its abstract methods `update_policy()` and `process_experience()`; assign your RL policy to `self.policy` in the class initialization; then override `prep_action()`, `get_state()`, `get_reward()`, `get_done()` or other methods for further customization (see `examples/fei/fei_agent.py`)

3. Implement a subclass of `RLServer` in `simple_rl_server.py` and its abstract methods `prep_state()` and `apply_action()`; then override `prep_state()` or other methods for further customization (see `examples/fei/fei_server.py`)

4. If you want to customize other components of FL, it will be the same as what you're supposed to do with plato; e.g., a customized client (see `examples/fei/fei_client.py`) or a customized trainer (see `examples/fei/fei_trainer.py`)

5. Implement a starting program that runs the DRL agent as a subprocess along with the FL processes (follow `examples/fei/fei.py` as a template and replace components as you need)

6. Implement the `xxx.yml` configure file as you need and change the `os.environ['config_file']` to the path of `xxx.yml` file in the starting program

## How to run your customized DRL agent for learning and controlling FL?

To start the training/testing of your customized DRL-controlled FL training, just simply run the starting program, e.g.,

```
python examples/fei/fei.py 
```


