import torch
import logging
import random
import time
import os
import csv

from torch.utils.data import Dataset
from plato.config import Config
from plato.trainers import basic
from scipy.spatial.distance import cdist
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ELRLoss(nn.Module):
    """ Early learning regularization loss from https://proceedings.neurips.cc/paper/2020/file/ea89621bee7c88b2c5be6681c8ef4906-Paper.pdf"""
    def __init__(self, num_examp=50000, num_classes=10, beta=0.7):
        # num_examp = 60000 for mnist; 50000 for cifar10
        super(ELRLoss, self).__init__()
        self.device = Config().device()
        self.target = torch.zeros(num_examp, num_classes).to(self.device) 
        self.beta = beta
        self.lamda = 0 #1 #7 # should read from configuration file.

    def forward(self, outputs, labels, indices):
        y_pred = F.softmax(outputs,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()

        self.target[indices] = self.beta * self.target[indices] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = F.cross_entropy(outputs, labels)
        elr_loss = ((1-(self.target[indices] * y_pred).sum(dim=1)).log()).mean()
        loss = ce_loss +  self.lamda * elr_loss
        
        return  loss

class LIDLoss(nn.Module):
    """Lid based soft labeling, which can be considered as a loss modification"""
    def __init__(self):
        super(LIDLoss, self).__init__()

    def forward(self, outputs, labels, alpha):
        # check embedings here 
        self.device = Config().device()
        torch.autograd.set_detect_anomaly(True)

        pred_labels = F.one_hot(torch.argmax(outputs, 1), num_classes=10)
        labels = F.one_hot(labels, num_classes=10)

        alpha = torch.tensor(alpha).to(self.device)
        y_new = (alpha * labels.T + (1. - alpha) * pred_labels.T).T
        outputs =  outputs / torch.sum(outputs, axis=-1, keepdims=True)
        outputs = torch.clamp(outputs, 1e-7, 1.0 - 1e-7)

        loss = -torch.mean(torch.sum(y_new * torch.log(outputs), axis=-1))
        #logging.info(f"loss: %s", loss)
        return loss

class IndexedDataSet(Dataset):
    """A toy trainer to test noisy data source."""
    def __init__(self, dataset) -> None:
        super().__init__()
        self._wrapped_dataset = dataset

    def __len__(self):
        return len(self._wrapped_dataset)

    def __getitem__(self, index):
        return (index, self._wrapped_dataset.__getitem__(index))

class Trainer(basic.Trainer):
    """A toy trainer to test noisy data source."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(model, callbacks)
        self.cache_root =  os.path.expanduser("~/.cache")
        self.server_id = os.getppid()
        self.current_round = None
        

    def train_model(self, config, trainset, sampler, **kwargs):
        """The default training loop when a custom training loop is not supplied."""
        batch_size = config["batch_size"]
        self.sampler = sampler
        tic = time.perf_counter()

        self.run_history.reset()

        self.train_run_start(config)
        self.callback_handler.call_event("on_train_run_start", self, config)

        # self.train_loader = self.get_train_loader(batch_size, trainset, sampler)
        self.train_loader = self.get_indexed_train_loader(batch_size, trainset, sampler)

        # Initializing the loss criterion
        self._loss_criterion = self.get_loss_criterion()

        # Initializing the optimizer
        self.optimizer = self.get_optimizer(self.model)
        self.lr_scheduler = self.get_lr_scheduler(config, self.optimizer)
        self.optimizer = self._adjust_lr(config, self.lr_scheduler, self.optimizer)
        
        
        self.model.to(self.device)
        self.model.train()

        total_epochs = config["epochs"]

        # Corrected label and indices
        # [[indices_1, labels_1], [indices_2, labels_2], ...]
        corrections = []
        corrected = True
        # lamda is a factor in calculating alpha
        self.lamda = self.current_round / config["rounds"] 
        

        for self.current_epoch in range(1, total_epochs + 1):
            self._loss_tracker.reset()
            self.train_epoch_start(config)
            self.callback_handler.call_event("on_train_epoch_start", self, config)
            lid_batch = []
            for batch_id, (indices, (examples, labels)) in enumerate(self.train_loader):
                self.train_step_start(config, batch=batch_id)
                self.callback_handler.call_event(
                    "on_train_step_start", self, config, batch=batch_id
                )

                examples, labels = examples.to(self.device), labels.to(self.device)

                loss,lid,correction= self.perform_forward_and_backward_passes(
                    config, examples, labels, indices)
                #)
                if len(correction) > 0:
                    corrections.extend(correction)
                    #logging.info(f"corrections: %s",corrections)
                # logging.info(f"lids for this batch is: %s", lids)
                lid_batch.extend(lid)
                #corrections = [refurbushable,new_labels]
               

                # label correction for selfie
                # if len(lid)>0:
                #    corrections.append(new_labels)

                # if not corrected:
                #    corrections.append(self.magic_label_correction(indices))

                self.train_step_end(config, batch=batch_id, loss=loss)
                self.callback_handler.call_event(
                    "on_train_step_end", self, config, batch=batch_id, loss=loss
                )

            corrected = True

            self.lr_scheduler_step()

            if hasattr(self.optimizer, "params_state_update"):
                self.optimizer.params_state_update()

            # Simulate client's speed
            if (
                self.client_id != 0
                and hasattr(Config().clients, "speed_simulation")
                and Config().clients.speed_simulation
            ):
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if (
                hasattr(Config().server, "request_update")
                and Config().server.request_update
            ):
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{self.current_epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)

            self.run_history.update_metric("train_loss", self._loss_tracker.average)
            self.train_epoch_end(config)
            self.callback_handler.call_event("on_train_epoch_end", self, config)

            # save lids locally
            file_path = './lids'+ str(self.client_id) +'.csv' 
            with open(file_path, "a") as file:
                file.write(str(np.mean(lid_batch)))
                file.write("\n")
            #logging.info(f"mean of lids this batch is: %.8lf", np.mean(lid_batch))

        self.save_pseudo_labels(corrections)

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)

    def magic_label_correction(self, indices):
        '''Update the labels at target positions'''
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices)
        logging.info(f"Correcting the label for %s", indices)
        # random correction
        pseudo_labels = torch.randint(0, 10, indices.size())
        return [indices, pseudo_labels]

    def save_pseudo_labels(self, corrections):
        # Organize corrected labels, corrections should be formatted as
        # [[indices_1, labels_1], [indices_2, labels_2], ...]
        #logging.info(f"we are saving labels: %s", corrections)
        if len(corrections) > 0: 
            if len(corrections) == 1: 
                indices = corrections[0][0]
                pseudo_labels = corrections[0][1]
            else: 
                indices = []
                pseudo_labels = []
                for x in corrections:
                    indices.append(x[0])
                    pseudo_labels.append(x[1])
                indices = torch.stack(indices)
                pseudo_labels = torch.stack(pseudo_labels)
                    
                #indices = torch.cat([x[0] for x in corrections]) 
                #pseudo_labels = torch.cat([x[1] for x in corrections])
            logging.info(f" [Client #{self.client_id}] Replaced labels at {indices} to {pseudo_labels}")
            # Dump pseudo labels to file
            label_file = f"{self.server_id}-client-{self.client_id}-label-updates.pt"
            label_file = os.path.join(self.cache_root, label_file)
            torch.save([indices, pseudo_labels], label_file)
        else: 
            logging.info(f"[Client #{self.client_id}] Keeps the label untouched.")


    def get_indexed_train_loader(self, batch_size, trainset, sampler):
        return torch.utils.data.DataLoader(
            dataset=IndexedDataSet(trainset), shuffle=False, batch_size=batch_size, sampler=sampler
        )
     
    # def get_loss_criterion(self):
    #     #Returns the regularization loss criterion.
    #     return LIDLoss() #ELRLoss()

    def perform_forward_and_backward_passes(self, config, examples, labels, indices):
        """
        Perform forward and backward passes in the training loop.

        Arguments:
        config: the configuration.
        examples: data samples in the current batch.
        labels: labels in the current batch.

        Returns: loss values after the current batch has been processed.
        """ 
        
        self.optimizer.zero_grad()
        #below is for lid##########################################################
        # a dict to store the activations
        # add a hook to get intermediate results
        
        activation = []
        def getActivation(name):
        # the hook signature
            def hook(model, input, output):
                activation.append(output.detach().cpu().numpy())
                # logging.info(f"in hook function: %s", activation)
            return hook 
        h1 = self.model.relu4.register_forward_hook(getActivation('relu4'))
        
        outputs = self.model(examples)
        
        lids = self.calculate_lid(activation) # try half
        # remove the hook
        h1.remove()

        # lid is for each sample
        # load min_lid from local file
        cache_root =  os.path.expanduser("~/.cache")
        lid_file = f"{self.server_id}-{self.client_id}-lids.pt"
        lid_file = os.path.join(cache_root, lid_file)

        # below is to study alpha changes at each sample
        alpha_file = f"{self.server_id}-{self.client_id}-alphas.pt"
        alpha_file = os.path.join(cache_root, alpha_file)

        if not os.path.exists(alpha_file):
            # Label file not exsits, create a new dic 
            alpha_dict = {}
        else: 
            # load min from file
            alpha_dict =  torch.load(alpha_file)


        if not os.path.exists(lid_file):
            # Label file not exsits, create a new dic 
            lid_dict = {}
        else: 
            # load min from file
            lid_dict =  torch.load(lid_file)

        #logging.info(f"indics: %s", indices)
        #min_lids = [lid_dict[index] for index in indices if index in lid_dict]
        alphas = []
        corrections = []
        #refurbushable = []
        #new_labels = []
        pred_outputs = torch.argmax(outputs, 1).tolist()
        #logging.info(f"pre_outputs: %s", pred_outputs)
        for lid, index, pre_label in zip(lids, indices,pred_outputs):
            #logging.info(f"index: %s", index.item())
            index = index.item()
            if index in lid_dict:

                # calculate 
                alpha = -np.power (1/8 * lid / np.min(lid_dict[index]) - 1/8, 2)+1
                #- 1/2 * np.exp(- 1 * lid / np.min(lid_dict[index])) * np.log(1/2*np.exp(- 1 * lid / np.min(lid_dict[index]))) # np.exp(-self.lamda * lid / np.min(lid_dict[index]))
                #logging.info(f"alpha: %s",alpha)
                #logging.info(f"min: %s", np.min(lid_dict[index]))
                alphas.append(alpha)

                # save alphas for study (should be remove later)
                alpha_dict[index].append(alpha)
                # threshold for label correction
                if alpha <= 0.95: # 0.95: 
                    corrections.append([torch.tensor(index),torch.tensor(pre_label)])
                    # refurbushable.append(index)
                    # new_labels.append(pre_label)
                    #logging.info(f"refurbushable: %s", refurbushable)
                    #logging.info(f"new_labels: %s", pre_label)

                # update lid_dict
                if lid < lid_dict[index]:
                    # replace with a smaller value
                    lid_dict[index] = lid
                # recording lids tractory for samples
                # this is to study lid changes on each sample
                # lid_dict[index].append(lid)
            else: 
                # add into lid_dict
                lid_dict[index] = lid #[lid]
                alpha_dict[index] = [1]
                alphas.append(1)

        #logging.info(f"alpha_dict: %s",alpha_dict)
        torch.save(alpha_dict,alpha_file)

        # refurbushable = []
        # loop over the dict and find index whose value is greater than a threshold
        # for index, value in data_dict.items():
        # # Check if the value is greater than the threshold
        # if value > threshold:
        #     # If it is, add the key to the list
        #     keys_above_threshold.append(key)
        # save lid_dict 
        torch.save(lid_dict, lid_file)

        # i = 0
        # for key, value in lid_dict.items():
        #     logging.info(f"key: %s, value: %s", key, value)
        #     i += 1
        #     if i >= 10:
        #         break
            # below is useless
            #expansion = lid / self.min_lid
            # should update and preserve the min_lid
            #alpha = np.exp(-self.lamda * expansion)
        
        ######above is for lid only #################################################################
        # this loss is for lid 
        #loss = self._loss_criterion(outputs, labels, alphas)

        # this is for cross-entropy by defualt
        loss = self._loss_criterion(outputs, labels)

        ###############below for selfie tracking
        """
        
        # outputs is the prediction, labels is the true lable
        # calculate f() value from past prediction
        #preds_file = "~/.cache/"+str(self.server_id) + "-" + str(self.client_id) +"preds.pt"
        cache_root =  os.path.expanduser("~/.cache")
        preds_file = f"{self.server_id}-{self.client_id}-preds.pt"
        preds_file = os.path.join(cache_root, preds_file)
        
        if not os.path.exists(preds_file):
            # Label file not exsits, create a new dic 
            preds_dict = {}
            
        else: 
            # load min from file
            preds_dict =  torch.load(preds_file)

        refurbushable = []
        f_eposilon = 0.05 # threshold {0.05, 0.10, 0.15, 0.20} from paper
        f_sigma = 25 # normalization
        pred_outputs = torch.argmax(outputs, 1)
        logging.info(f"pred_outputs: %s", pred_outputs)

        for output, index in zip(pred_outputs, indices):
            index = index.item()
            if index in preds_dict:
                # append new pred into dict
                preds_dict[index].append(output.detach().cpu().numpy())
                logging.info(f"outputs: %s", output)

                # calculate f_value
                values, counts = np.unique(preds_dict[index], return_counts=True)
                logging.info(f"counts: %s", counts)
                f_value = 0
                for count in counts: 
                    f_value -= count / len(preds_dict[index]) * np.log(count / len(preds_dict[index]))
                f_value /= f_sigma
                logging.info(f"f_value here: %s", f_value)
                #logging.info(f"f_eposilon: %s", f_eposilon)
                if f_value < f_eposilon:
                    refurbushable.append(index)

            else: 
                # add into lid_dict
                preds_dict[index] = [output.detach().cpu().numpy()]
                
        # save lid_dict 
        torch.save(preds_dict, preds_file)
        # the output of the selfie is the refurbushable list that contains indices of samples that can be modified.
        """
        #############################################


        # this loss is for regularization
        #loss = self._loss_criterion(outputs, labels)#, indices)
        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.step()
        #logging.info(f"refurbushable: %s", refurbushable)
        return loss, lids, corrections #refurbushable, new_labels #lids #refurbushable 
     

    def calculate_lid(self, batch_data, k=40):
        # calculate lids for samples in current batch
        # k is for k-nearest neighbours 
        #logging.info(f"type of batch_data: %s", type(batch_data))
        batch_data = np.stack(batch_data, axis=0)

        batch = batch_data.reshape((batch_data[0].shape[0], -1))#.cpu()
        batch = np.asarray(batch, dtype=np.float32)

        k = min(k, len(batch) - 1) # number of neighbours

        a = cdist(batch, batch)
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]

        f = lambda v: - k / np.sum(np.log(v / v[-1] + 1e-8)) 
        lids = np.apply_along_axis(f, axis=1, arr=a)
        #logging.info(f"lids: %s", lids)
        
        return lids


