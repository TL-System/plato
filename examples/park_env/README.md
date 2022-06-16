# Park

### Salma June 8: Exampe of running A2C on abr_sim 

`python a2c_abr_sim.py`

### Trace files from GENET project:

`park/envs/abr_sim/Genet/`

### Real system interface
```python
import park
import agent_impl  # implemented by user

env = park.make('congestion_control')

# the run script will start the real system
# and periodically invoke agent.get_action
agent = Agent(env.observation_space, env.action_space)
env.run(agent)
```

The `agent_impl.py` should implement
```python
class Agent(object):
    def __init__(self, state_space, action_space, *args, **kwargs):
        self.state_space = state_space
        self.action_space = action_space

    def get_action(self, obs, prev_reward, prev_done, prev_info):
        act = self.action_space.sample()
        # implement real action logic here
        return act
```

### Simulation interface
Similar to OpenAI Gym interface.
```python
import park

env = park.make('load_balance')

obs = env.reset()
done = False

while not done:
    # act = agent.get_action(obs)
    act = env.action_space.sample()
    obs, reward, done, info = env.step(act)
```

### Contributors

| Environment                     | env_id                         | Committers                       |
| ------------------------------- | ------------------------------ | -------------------------------- |
| Adaptive video streaming        | abr, abr_sim                   | Hongzi Mao, Akshay Narayan       |
| Spark cluster job scheduling    | spark, spark_sim               | Hongzi Mao, Malte Schwarzkopf    |
| SQL database query optimization | query_optimizer                | Parimarjan Negi                  |
| Network congestion control      | congestion_control             | Akshay Narayan, Frank Cangialosi |
| Network active queue management | aqm                            | Mehrdad Khani, Songtao He        |
| Tensorflow device placement     | tf_placement, tf_placement_sim | Ravichandra Addanki              |
| Circuit design                  | circuit_design                 | Hanrui Wang, Jiacheng Yang       |
| CDN memory caching              | cache                          | Haonan Wang, Wei-Hung Weng       |
| Multi-dim database indexing     | multi_dim_index                | Vikram Nathan                    |
| Account region assignment       | region_assignment              | Ryan Marcus                      |
| Server load balancing           | load_balance                   | Hongzi Mao                       |
| Switch scheduling               | switch_scheduling              | Ravichandra Addanki, Hongzi Mao  |

### Misc
Note: to use `argparse` that is compatiable with park parameters, add parameters using
```python
from park.param import parser
parser.add_argument('--new_parameter')
config = parser.parse_args()
print(config.new_parameter)
```
