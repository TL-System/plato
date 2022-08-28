## Paramter meanings in the configuration files
### Already exisiting documentation
Many of the configuration file parameters and their meanings can be found in the configuration tab [here](https://platodocs.netlify.app/configuration.html)

### Clients
Everything in here can be answered in the already exisiting documentation above
### Server
```percentile:``` This is the initial percentile of clients the server will choose to aggregate in the first round \

```percentile_aggregate:``` There are seven different options for this parameter and it is used for different difficulty score metrics
1. ```False``` When false, there is no curriculum learning being used
2. ```actor_loss``` This uses the loss of the actor
3. ```critic_loss``` This uses the loss of the critic
4. ```actor_grad``` This uses the rate at which the actor loss is decreasing
5. ```critic_grad``` This uses the rate at which the critic loss is decreasing
6. ```sum_actor_fisher``` This uses the sum of the diagonals of the fisher information matrix of the actor
7. ```sum_critic_fisher``` This uses the sum of the diagonals of the fisher information matrix of the critic \

```percentile_increase:``` This is the amount that the server gradually increases the percentile every round to finally reach the 100th percentile

```mul_fisher:``` True or False, when true.. the server uses interference avoidance for the server, when false it does not

The other parameters can be answered in the already exisiting documentation above 

### Data
No parameters in this section are actually used in reinforcement learning. However, the reason there are parameters here that exist in the configuration file is due to the fact that Plato requires these to run otherwise there will be a compilation error

### Trainer:
```manual_seed:``` This seed is used and set at the start to maintain reproducibility
```penalize_omega:``` True or False, when true.. the clients use interference avoidance, when false they do not \
```lamda:``` This factor is the factor used to avoid interference. Please note that lamda has a factor of 2 so if lamda is 2 then it's actually one
**Important Note:** If running multiple jobs say on compute canada or locally at the same time, please make sure that ```max_concurrency``` is >= to the number in ```per_round``` back in the clients section.

The other parameters can be answered in the already exisiting documentation above 

### Algorithm:
```mode:``` Set it to train to train clients \
```gamma:``` Reward discounted factor \
```learning_rate:``` The learning rate used in FedADP \
```eval_freq:``` Frequency which to evaluate the policy \
```batch_size:``` batch size to be sampled from memory \
``` entropy_ratio:``` \
``` entropy_decay:``` \
``` entropy_min:``` These and the two above are all entropy ratio hyperparameters \
```grad_clip_val:``` grad clip hyperparamter \
```max_round_episodes:``` The number of episodes to run each round
```difficulty_levels:```
```traces_per_task:``` This and the difficulty level above are used for curriculum learning **elaborate more?** \
```save_models:``` save models or no? True for save, False for no \
```env_name:``` The name of environment \
```env_park_name:``` The park environment name \
```algorithm_name:``` The name of the algorithm

### Results:
```results_dir:``` The path of the results directory \
```file_name:``` The name of the files \
```seed_random_path:``` The path of the seed folders

The other parameters can be answered in the already exisiting documentation above

## Different parameter configurations for each respective algorithm

### Cascade (curriculum learning with interference avoidance)
```percentile_aggregate: critic_grad```\
```mul_fisher: False``` \
```penalize_omega: True``` \
```lamda: 2```

### Curriculum Learning without interference avoidance
```percentile_aggregate: critic_grad```\
```mul_fisher: False``` \
```penalize_omega: False``` \
```lamda: does not matter since penalize omega is false```

### Interference avoidance for clients but without curriculum learning (called MAS)
```percentile_aggregate: False```\
```mul_fisher: False``` \
```penalize_omega: True``` \
```lamda: 2```

### Interference avoidance for the server but without curriculum learning (uses mul_fisher)
```percentile_aggregate: Does not matter```\
```mul_fisher: True``` \
```penalize_omega: Does not matter``` \
```lamda: Does not matter```

### FedAvg
```percentile_aggregate: False```\
```mul_fisher: False``` \
```penalize_omega: False``` \
```lamda: does not matter since penalize omega is false```

### FedADP
```percentile_aggregate: False```\
```mul_fisher: False``` \
```penalize_omega: False``` \
```lamda: does not matter since penalize omega is false``` \
**Note:** Be sure to run the a2cadp.py file instead of a2c.py to run FedAdp as it uses a different server aggregation algorithm

### Preset Curriculum
```percentile_aggregate: Does not matter```\
```mul_fisher: Does not matter``` \
```penalize_omega: False``` \
```lamda: does not matter since penalize omega is false``` \
**Change source code in** ```async def federated_averaging(self, updates):``` in the server file to select specific clients

**If one wants to test out the different difficulty score metrics for the curriculum, then you change the paramter ```percentile_aggregate``` to the five other options**



## What to change in configuration files to run multiple jobs

* Make sure these two paths have the specific algorithm name with their respective seed number appended to it

```shell 
checkpoint_path
model_path
```
* For example, if we wanted to run both FedAvg seed 10 and FedAvg seed 15, we would have _two_ configuration files and they would have 
```checkpoint_path: checkpoints_fedavg_aggregate_seed_10``` | ```model_path: models/pretrained_fedavg_aggregate_seed_10```
```checkpoint_path: checkpoints_fedavg_aggregate_seed_15``` | ```model_path: models/pretrained_fedavg_aggregate_seed_15``` respectively
* Make sure port numbers are also different for every file different
