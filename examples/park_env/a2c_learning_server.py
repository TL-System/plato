"""
A federated learning server using federated averaging to train Actor-Critic models.

To run this example:

python examples/customized/custom_server.py -c examples/customized/server.yml
"""

import logging
import asyncio
from multiprocessing.sharedctypes import Value
from plato.servers import fedavg
from plato.config import Config
import pickle
import random
import numpy as np
import csv

class A2CServer(fedavg.Server):
    """ Federated learning server using federated averaging to train Actor-Critic models. """
    """ A custom federated learning server. """

    def __init__(self, algorithm_name, env_name, model = None, trainer = None, algorithm = None):
        super().__init__(trainer = trainer, algorithm = algorithm, model = model)
        self.algorithm_name = algorithm_name
        self.env_name = env_name
        logging.info("A custom server has been initialized.")
        
    async def federated_averaging(self, updates):
        """Aggregate weight updates from the clients using federated averaging."""

        weights_received = self.compute_weight_deltas(updates)

        actor_loss_list = []
        critic_loss_list = []

        critic_percentile_aggregation = None
        actor_percentile_aggregation = None

        if Config().server.percentile_aggregate:
            
            print("-----------------------")
            print("WE ARE AGGREGATING BASED ON PERCENTILE!")
            print("-----------------------")

            percentile = Config().server.percentile

            self.error_check()
            
            #Create lists and get threshold for percentile aggregation
            if Config().server.actor_loss_aggregate:
                actor_loss_list = self.create_loss_lists(True, updates)
                actor_percentile_aggregation = np.percentile(np.array(actor_loss_list), percentile)
                print("Actor percentile is ", actor_percentile_aggregation)
                print("Actor loss list: ", actor_loss_list)
            
            if Config().server.critic_loss_aggregate:
                critic_loss_list = self.create_loss_lists(False, updates)
                critic_percentile_aggregation = np.percentile(np.array(critic_loss_list), percentile)
                print("Critic percentile is ", critic_percentile_aggregation)
                print("Critic loss list: ", critic_loss_list)
            actor_path = Config().results.results_dir +"/"+Config().results.file_name+"_percentile_actor_loss"
            critic_path =  Config().results.results_dir +"/"+Config().results.file_name+"_percentile_critic_loss"

            #save percentile to files
            #one of the lists has to have something, else error raised
            #if critic list has nothing we aggregate actor
            if len(critic_loss_list) == 0:
                self.save_files(actor_path, actor_percentile_aggregation)
            else:
                self.save_files(critic_path, critic_percentile_aggregation)
        else:
            print("-----------------------")
            print("WE ARE NOT AGGREGATING BASED ON PERCENTILE!")
            print("-----------------------")
          
        # Perform weighted averaging for both Actor and Critic
        actor_avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0][0].items()
        }
        critic_avg_update = {
            name: self.trainer.zeros(weights.shape)
            for name, weights in weights_received[0][1].items()
        }

        for i, update in enumerate(weights_received):
            print("Client# ", i)
            __, report, __, __ = updates[i]
            actor_loss = report.actor_loss
            critic_loss = report.critic_loss
            client_id = report.client_id
            client_path =  Config().results.results_dir +"/"+Config().results.file_name+"_client_saved"

            update_from_actor, update_from_critic = update

            if Config().server.percentile_aggregate == False:
                for name, delta in update_from_actor.items():
                    actor_avg_update[name] += delta * 1.0/6.0

                for name, delta in update_from_critic.items():
                    critic_avg_update[name] += delta * 1.0/6.0
            else:
                # TODO: weight of delta should be the weightage of the client 1/number of clients choosen to be aggreggateed
                if Config().server.actor_loss_aggregate and actor_loss >= actor_percentile_aggregation:
                    print("\n-----------------------")
                    print("AGGREGATING ONLY BASED ON ACTOR LOSS")
                    print("-----------------------\n")
                    print("Client %s is choosen" % str(client_id))
                    self.save_files(client_path, client_id)
                    for name, delta in update_from_actor.items():
                        actor_avg_update[name] += delta * 2.0/6.0
                    for name, delta in update_from_critic.items():
                        critic_avg_update[name] += delta * 2.0/6.0
                
                if Config().server.critic_loss_aggregate and critic_loss >= critic_percentile_aggregation:
                    print("\n-----------------------")
                    print("AGGREGATING ONLY BASED ON CRITIC LOSS")
                    print("-----------------------\n")
                    print("Client %s is choosen" % str(client_id))
                    self.save_files(client_path, client_id)
                    for name, delta in update_from_actor.items():
                        actor_avg_update[name] += delta * 2.0/6.0
                    for name, delta in update_from_critic.items():
                        critic_avg_update[name] += delta * 2.0/6.0
            # Yield to other tasks in the server
            await asyncio.sleep(0)
        
        return actor_avg_update, critic_avg_update

    def save_files(self, file_path, data):
        #To avoid appending to existing files, if the current roudn is one we write over
        if self.current_round == 1:
            with open(file_path+".csv", 'w') as filehandle:
                writer = csv.writer(filehandle)
                writer.writerow([data])
        else:
            with open(file_path+".csv", 'a') as filehandle:
                writer = csv.writer(filehandle)
                writer.writerow([data])


    def create_loss_lists(self, actor_passed, updates):
        """Creates the lost lists"""
        loss_list = []
        for (_, report, _, _) in updates:
            if actor_passed:
                loss_list.append(report.actor_loss)
            else:
                loss_list.append(report.critic_loss)

        return loss_list

    def error_check(self):
        if Config().server.actor_loss_aggregate and Config().server.critic_loss_aggregate:
            raise ValueError("CANNOT AGGREGATE ACTOR AND CRITIC LOSS AT THE SAME TIME, please make one false and one true")

        if Config().server.actor_loss_aggregate == False and Config().server.critic_loss_aggregate == False:
            raise ValueError("Why percentile aggregate if you don't want to aggregate either actor or critic? Please make one of them true")


    def save_to_checkpoint(self):
        """ Save a checkpoint for resuming the training session. """
        checkpoint_path = Config.params['checkpoint_path']

        copy_algorithm = self.algorithm_name
        if '_' in copy_algorithm:
            copy_algorithm= copy_algorithm.replace('_', '')
        
        env_algorithm = self.env_name+copy_algorithm
        model_name = env_algorithm
        if '_' in model_name:
            model_name.replace('_', '')
        filename = f"checkpoint_{model_name}_{self.current_round}.pth"
        logging.info("[%s] Saving the checkpoint to %s/%s.", self,
                     checkpoint_path, filename)
        self.trainer.save_model(filename, checkpoint_path)
        self.save_random_states(self.current_round, checkpoint_path)

        # Saving the current round in the server for resuming its session later on
        with open(f"{checkpoint_path}/current_round.pkl",
                  'wb') as checkpoint_file:
            pickle.dump(self.current_round, checkpoint_file)