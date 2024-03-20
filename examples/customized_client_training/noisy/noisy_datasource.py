import os
import random
import torch
import logging

import plato.datasources.registry as data_registry
import plato.trainers.registry as trainers_registry
import plato.models.registry as model_registry
from plato.datasources import base
from plato.config import Config

class NoisyDataSource(base.DataSource):
    """A custom datasource with custom training and validation datasets."""

    def __init__(self):
        super().__init__()

        # Save the process id of the server
        self.cache_root =  os.path.expanduser("~/.cache")
        self.server_id = os.getppid()

        self._wrapped_datasource = data_registry.get()
        noise_engine = NoiseEngine()
        noise_engine.add_noise(self._wrapped_datasource)

        self.trainset = self._wrapped_datasource.trainset
        self.testset = self._wrapped_datasource.testset
        
        # Save the initial noisy targets
        self.init_noisy_targets = self.trainset.targtes
    
    def num_train_examples(self):
        return self._wrapped_datasource.num_train_examples()

    def num_test_examples(self):
        return self._wrapped_datasource.num_test_examples()



class NoiseEngine:
    def __init__(self) -> None:
        self.device = None
        self.cache_root =  os.path.expanduser("~/.cache")
        self.dataset_name = Config().data.datasource
        self.noise_model = Config().data.noise.noise_model

        self.noise_accuracy = 0.8 if not hasattr(Config().data.noise, "noise_accuracy") else Config().data.noise.noise_accuracy
        self.top_k = 5 if not hasattr(Config().data.noise, "top_k") else Config().data.noise.top_k
        self.noise_ratio = 0.4 if not hasattr(Config().data.noise, "noise_ratio") else Config().data.noise.noise_ratio

        self.random_seed = 1 if not hasattr(Config().data, "random_seed") else Config().data.random_seed

    def add_noise(self, datasource):
        if not os.path.exists(self.cache_root):
            os.mkdir(self.cache_root)

        nosie_file = f"{self.dataset_name}-{self.noise_model}-{self.top_k}-label.pt"
        nosie_file = os.path.join(self.cache_root, nosie_file)

        # Label file not exsits
        if not os.path.exists(nosie_file):
            logging.info("Noise label not exists, generating...")
            self.generate_noise_label(datasource, nosie_file)

        topK_predicts = torch.load(nosie_file) 
        self.replace_labels(datasource, topK_predicts)

    def generate_noise_label(self, datasource, nosie_path):
        model = model_registry.get(model_name = self.noise_model)
        ckpt_file = f"{self.dataset_name}-{self.noise_model}-{self.noise_accuracy}-ckpt.pt"
        ckpt_file = os.path.join(self.cache_root, ckpt_file)

        # Pre-trained weights not exist
        if not os.path.exists(ckpt_file):
            logging.info("Model checkpoint not exists, training from scratch.")
            self.train_model(model=model, trainset=datasource.trainset, ckpt_path=ckpt_file)

        model.load_state_dict(torch.load(ckpt_file), strict=True)

        topK_predicts = self.predict(model, datasource.trainset, self.top_k)
        torch.save(topK_predicts, nosie_path)

    def predict(self, model, trainset, k):
        dataloader = torch.utils.data.DataLoader(
                dataset=trainset, shuffle=False, batch_size=64
        )
        model.to(self.device)
        model.eval()
        topk_pred = []
        with torch.no_grad():
            for batch_id, (examples, labels) in enumerate(dataloader):
                examples, labels = examples.to(self.device), labels.to(self.device)
                outputs = model(examples)
                _, pred =torch.topk(outputs, k, dim=1)
                topk_pred.append(pred)

        topk_pred = torch.vstack(topk_pred).clone().detach().to("cpu")
        return topk_pred

    def train_model(self, model, trainset, ckpt_path):
        config = Config().trainer._asdict()

        train_loader = torch.utils.data.DataLoader(
                dataset=trainset, shuffle=True, batch_size=config["batch_size"]
        )
        
        trainer = trainers_registry.get()
        self.device = trainer.device

        # Initializing the loss criterion
        _loss_criterion = trainer.get_loss_criterion()

        # Initializing the optimizer
        optimizer = trainer.get_optimizer(model)
        lr_scheduler = trainer.get_lr_scheduler(config, optimizer)
        optimizer = trainer._adjust_lr(config, lr_scheduler, optimizer)

        model.to(self.device)
        model.train()

        epoch_counter = 1
        while True:
            correct = 0
            total = 0
            for batch_id, (examples, labels) in enumerate(train_loader):
                examples, labels = examples.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(examples)

                # Backward
                loss = _loss_criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if hasattr(optimizer, "params_state_update"):
                optimizer.params_state_update()

            # Check training accuracy
            train_acc = correct / total
            if train_acc >= self.noise_accuracy:
                logging.info(f"Epoch: {epoch_counter} training accuracy: {train_acc}, target nosie accuracy ({self.noise_accuracy}) achieved.")
                break
            else:
                logging.info(f"Epoch: {epoch_counter} training accuracy: {train_acc}.")

            epoch_counter += 1
                
        # Save the trained model in cache
        torch.save(model.state_dict(), ckpt_path)

    def replace_labels(self, datasource, topK_predicts):
        if not isinstance(datasource.trainset.targets, torch.Tensor):
            datasource.trainset.targets = torch.tensor(datasource.trainset.targets)
        
        top1_accuracy = sum(topK_predicts[:,0].to("cpu") == datasource.trainset.targets) / len(datasource.trainset.targets)
        logging.info(f"Top 1 accuracy from noise model predictions: {top1_accuracy}")
        total_nums = len(datasource.trainset)
        noise_nums = int(total_nums * self.noise_ratio) 

        # Select indices to replace labels
        random.seed(self.random_seed)
        nosie_indices = random.sample(range(total_nums), noise_nums)

        # Choose a label from TopK predicts
        rand_topK = torch.tensor(random.choices(range(1, self.top_k), k=noise_nums))
        offset = torch.arange(0, noise_nums).long() * self.top_k

        selected_predicts = topK_predicts[nosie_indices].flatten()
        selected_predicts = selected_predicts[offset + rand_topK].detach().clone().to("cpu")
        datasource.trainset.targets[nosie_indices] = selected_predicts
        logging.warning(f"Replaced {noise_nums} labels with noisy predctions.")
