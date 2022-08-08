# How A2C is being ran on Plato
A2C utilizes a custom model, trainer, algorithm, client, and server. \
There is already lots of information on creating your own custom training in the plato documentation and it is useful to reference these documents to get started
## Getting started
Take a look at ```a2c.py``` for an example of how to pass in the different classes. Notice that model, algorithm, trainer have their classes and not objects passed into the client. This is because the plato will have each client create their own respective algorithm, trainer, and model objects at initialization. 

## Model file (a2c.learning_model.py)
A2C utilizes an actor-critic model. So we would need our own custom model that has both actor and critic
inside it. This is done by creating a ```wrapper class``` that will hold both actor and critic models instantiated inside it as members. Plato also requires ```def cpu```, ```def to```, ```def eval```,  ```def train``` to be implemented as well. When implementing these, be sure to implement one for the actor and one for the critic. Implement a ```static method``` as well that returns an instantiation of the model class. 
The model class also uses the same initial untrained model every time from untrained directory underneath models.
## Trainer file (a2c_learning_trainer.py)
This is where the RL algorithm A2C is actually being used by the clients to do reinforcement learning. This is where you would want to **create the environment** from park inside the constructor. The constructor should also call the parent class's constructor and pass in the model to create an instance of the model.
### Reproducibility
To be able to compare different algorithms such as FedADP vs FedAvg vs Cascade, we would need some sort of reproducibility so that the experiments are not entirely different. The way we achieved this was by setting a manual seed at the start of training inside ```train_model```. At the end of each training round, we would then save the seeds by calling ```save_seeds()```. On the next round, we call ```restore_seeds()``` to resume from the same session. This gives some sort of reproducibility between algorithms.

### What needs to be saved and loaded each round.
The methods ```save_model()``` and ```load_model()``` are very important. Save model saves the model state dicts at the end of each round and load_model will load these state dicts at the start of the next round. They should be saving and loading from the same path. \
**Note:** Client id == 0 refers to the server

## Client file (a2c_learning_client.py)
This file is really short and it only servers one purpose and that is to communicate and report back certain metrics to the server. The constructor should call the parent class's constructor and pass in our custom algorithm, model, and trainer. 
<hr>
This file also has a report class that is a dataclass with members that we want to report back to the server. Inside the ```train``` method, we return a report with all the members of the data class filled out. We get the loss, grad, and fisher by calling their load respective load methods implemented in the trainer class. 
<hr>
These load methods will load the loss, grad, and fisher saved into a file from the trainer class. This is how we should load these metrics due to the fact that the client and trainer files can't communicate with one another (getters that return the value actually do not return anything besides 0 because they are reset back to 0 once we are ready to return information to the server).

## Server file (a2c_learning_server.py)
This file is used to aggregate the clients and their models. The constructor should call it's parent's class constructor and pass in the custom algorithm, trainer, and model. 
<hr>
The method ```async def federated_averaging(self, updates):``` first computes the weight deltas implemented in the algorithm file and then checks if we want to use classic federated averaging or what Cascade uses based on what the parameter ```percentile_aggregate``` is in the configuration file. If there is percentile aggregate, it will ```select_metric```  as well as ```create_loss_list``` based on which parameter it is. <hr>
This file also saves checkpoints at the end of each round using the ```save_to_checkpoint``` method.

## Algorithm file (a2c_learning_algorithm.py)
This file basically deals with all the weights of the models returned back to the server. It extracts baseline weights that already exist and then computes the change in the incoming weights and returns this change (deltas) back to the server.