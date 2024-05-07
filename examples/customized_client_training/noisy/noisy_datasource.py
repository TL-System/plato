import os
import csv
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
        self.cache_root = os.path.expanduser("~/.cache")
        self.server_id = os.getppid()

        self._wrapped_datasource = data_registry.get()
        self.init_clean_targets = self.clone_labels(
            self._wrapped_datasource.trainset.targets
        )

        noise_engine = NoiseEngine()
        noise_engine.add_noise(self._wrapped_datasource)

        self.trainset = self._wrapped_datasource.trainset
        self.testset = self._wrapped_datasource.testset

        # Save the initial noisy targets
        self.init_noisy_targets = self.clone_labels(self.trainset.targets)

    def num_train_examples(self):
        return self._wrapped_datasource.num_train_examples()

    def num_test_examples(self):
        return self._wrapped_datasource.num_test_examples()

    def clone_labels(self, targets):
        if isinstance(targets, torch.Tensor):
            return targets.clone().detach()
        elif isinstance(targets, list):
            return torch.tensor(targets)
        else:
            raise TypeError(f"Cannot handle targets type: {type(targets)}")

    def read_cumulative_pseudo_labels(self, client_id):
        # Read client's pseudo label from file, return initial
        # noisy labels if file not exists.

        label_file = f"{self.server_id}-client-{client_id}-labels.pt"
        label_file = os.path.join(self.cache_root, label_file)

        if isinstance(self.init_noisy_targets, torch.Tensor):
            init_noisy_labels = self.init_noisy_targets.clone().detach()
        else:
            init_noisy_labels = torch.tensor(self.init_noisy_targets)

        try:
            [indices, pseudo_labels] = torch.load(label_file)
            init_noisy_labels[indices] = pseudo_labels
            logging.info(
                f"Client [{client_id}] Read pseduo labels {pseudo_labels} at indices {indices} from file."
            )
        except:
            logging.warning(
                "Client pseudo label file not exists, using initial noisy labels."
            )

        return init_noisy_labels

    def read_last_round_label_updates(self, client_id):
        label_file = f"{self.server_id}-client-{client_id}-label-updates.pt"
        label_file = os.path.join(self.cache_root, label_file)

        try:
            [indices, pseudo_labels] = torch.load(label_file)
            return [indices.flatten(), pseudo_labels.flatten()]
        except:
            logging.warning("Client pseudo labels not updated at last round.")
            return [[], torch.Tensor([])]

    def merge_label_updates(self, client_id):
        cumulative_pseudo_labels = self.read_cumulative_pseudo_labels(client_id)
        [updated_indices, updated_pseudo_labels] = self.read_last_round_label_updates(
            client_id
        )

        if len(updated_indices):

            cumulative_pseudo_labels[updated_indices] = updated_pseudo_labels

            cumulative_updated_indices = (
                (cumulative_pseudo_labels != self.init_noisy_targets)
                .nonzero()
                .flatten()
            )
            cumulative_updated_labels = cumulative_pseudo_labels[
                cumulative_updated_indices
            ]

            label_file = f"{self.server_id}-client-{client_id}-labels.pt"
            label_file = os.path.join(self.cache_root, label_file)
            torch.save(
                [cumulative_updated_indices, cumulative_updated_labels], label_file
            )

            # Remove update file to avoid duplicate eval
            self.remove_label_update_file(client_id)

    def setup_client_datasource(self, client_id):
        # Repalce the labels in trainset with client's pseudo labels
        self.merge_label_updates(client_id)
        self.trainset.targets = self.read_cumulative_pseudo_labels(client_id)

    def eval_pseudo_acc(self, client_id, client_indices):
        # Eval the cumulative pseudo labels
        cumulative_pseudo_labels_all = self.read_cumulative_pseudo_labels(client_id)
        cumulative_pseudo_labels = cumulative_pseudo_labels_all[client_indices]
        clean_labels_all = self.init_clean_targets[client_indices]

        clean_sample_nums_total = sum(cumulative_pseudo_labels == clean_labels_all)
        noisy_sample_nums_total = sum(cumulative_pseudo_labels != clean_labels_all)

        # Eval the last round modified labels
        [updated_indices, updated_pseudo_labels] = self.read_last_round_label_updates(
            client_id
        )
        noisy_labels = cumulative_pseudo_labels_all[updated_indices]
        clean_labels = self.init_clean_targets[updated_indices]

        label_modified_nums_this_round = len(updated_indices)
        # label_modified_nums_this_round = sum(updated_pseudo_labels != noisy_labels)

        clean_label_modified_nums_this_round = sum(
            torch.logical_and(
                updated_pseudo_labels != noisy_labels, noisy_labels == clean_labels
            )
        )

        clean_label_remained_nums_this_round = sum(
            torch.logical_and(
                updated_pseudo_labels == clean_labels, noisy_labels == clean_labels
            )
        )

        correct_noisy_relabel_nums_this_round = sum(
            torch.logical_and(
                clean_labels != noisy_labels,
                updated_pseudo_labels == clean_labels,
            )
        )

        wrong_noisy_relabel_nums_this_round = sum(
            torch.logical_and(
                clean_labels != noisy_labels,
                updated_pseudo_labels != clean_labels,
            )
        )

        # Test code
        if len(updated_indices):
            cumulative_pseudo_labels_all[updated_indices] = updated_pseudo_labels
        cumulative_pseudo_labels = cumulative_pseudo_labels_all[client_indices]
        applied_clean_sample_nums_total = sum(cumulative_pseudo_labels == clean_labels_all)

        if len(updated_indices):
            if len(set(updated_indices.tolist()) - set(client_indices)) > 0:
                print(123123)

        if applied_clean_sample_nums_total != clean_sample_nums_total + correct_noisy_relabel_nums_this_round - clean_label_modified_nums_this_round:
            print(123123)

        # Save the stats to csv
        stats = {
            "clean_sample_nums_total": clean_sample_nums_total,
            "noisy_sample_nums_total": noisy_sample_nums_total,
            "label_modified_nums_this_round": label_modified_nums_this_round,
            "clean_label_modified_nums_this_round": clean_label_modified_nums_this_round,
            "clean_label_remained_nums_this_round": clean_label_remained_nums_this_round,
            "correct_noisy_relabel_nums_this_round": correct_noisy_relabel_nums_this_round,
            "wrong_noisy_relabel_nums_this_round": wrong_noisy_relabel_nums_this_round,
        }
        stats = {
            k: v.item() if isinstance(v, torch.Tensor) else v for k, v in stats.items()
        }

        self.write_eval_csv(client_id, stats)

    def write_eval_csv(self, client_id, stats):
        root_folder = "./eval_pseudo"
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)

        client_folder = os.path.join(root_folder, str(self.server_id))
        if not os.path.exists(client_folder):
            try:
                os.mkdir(client_folder)
            except FileExistsError:
                pass

        csv_file = os.path.join(client_folder, f"pseudo-{client_id}.csv")

        if not os.path.exists(csv_file):
            with open(csv_file, "w") as f:
                w = csv.DictWriter(f, stats.keys())
                w.writeheader()
                w.writerow(stats)
        else:
            with open(csv_file, "a") as f:
                w = csv.DictWriter(f, stats.keys())
                w.writerow(stats)

    def remove_label_update_file(self, client_id):
        label_file = f"{self.server_id}-client-{client_id}-label-updates.pt"
        label_file = os.path.join(self.cache_root, label_file)
        if os.path.exists(label_file):
            os.remove(label_file)

    def targets(self):
        return self.init_noisy_targets

class NoiseEngine:
    def __init__(self) -> None:
        self.device = None
        self.cache_root = os.path.expanduser("~/.cache")
        self.dataset_name = Config().data.datasource
        self.noise_model = (
            None
            if not hasattr(Config().data.noise, "noise_model")
            else Config().data.noise.noise_model
        )

        self.noise_accuracy = (
            0.8
            if not hasattr(Config().data.noise, "noise_accuracy")
            else Config().data.noise.noise_accuracy
        )
        self.top_k = (
            5
            if not hasattr(Config().data.noise, "top_k")
            else Config().data.noise.top_k
        )
        self.noise_ratio = (
            0.4
            if not hasattr(Config().data.noise, "noise_ratio")
            else Config().data.noise.noise_ratio
        )

        self.random_seed = (
            1
            if not hasattr(Config().data, "random_seed")
            else Config().data.random_seed
        )

        self.cifar10N = (
            False
            if not hasattr(Config().data.noise, "cifar10N")
            else True
        )
        self.cifar10N_path = None
        self.cifar10N_split = None
        if self.cifar10N:
            self.cifar10N_path = Config().data.noise.cifar10N.path
            self.cifar10N_split = Config().data.noise.cifar10N.split
            assert self.cifar10N_split in ["worst", "aggre"]


    def add_noise(self, datasource):

        if self.cifar10N:
            logging.warn("Replace labels with CIFAR10N dataset.")
            noise_label = torch.load(self.cifar10N_path)[f'{self.cifar10N_split}_label']
            datasource.trainset.targets = noise_label.tolist()
            return
        
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
        model = model_registry.get(model_name=self.noise_model)
        ckpt_file = (
            f"{self.dataset_name}-{self.noise_model}-{self.noise_accuracy}-ckpt.pt"
        )
        ckpt_file = os.path.join(self.cache_root, ckpt_file)

        # Pre-trained weights not exist
        if not os.path.exists(ckpt_file):
            logging.info("Model checkpoint not exists, training from scratch.")
            self.train_model(
                model=model, trainset=datasource.trainset, ckpt_path=ckpt_file
            )

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
                _, pred = torch.topk(outputs, k, dim=1)
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
                logging.info(
                    f"Epoch: {epoch_counter} training accuracy: {train_acc}, target nosie accuracy ({self.noise_accuracy}) achieved."
                )
                break
            else:
                logging.info(f"Epoch: {epoch_counter} training accuracy: {train_acc}.")

            epoch_counter += 1

        # Save the trained model in cache
        torch.save(model.state_dict(), ckpt_path)

    def replace_labels(self, datasource, topK_predicts):
        if not isinstance(datasource.trainset.targets, torch.Tensor):
            datasource.trainset.targets = torch.tensor(datasource.trainset.targets)

        top1_accuracy = sum(
            topK_predicts[:, 0].to("cpu") == datasource.trainset.targets
        ) / len(datasource.trainset.targets)
        logging.info(f"Top 1 accuracy from noise model predictions: {top1_accuracy}")
        total_nums = len(datasource.trainset)
        noise_nums = int(total_nums * self.noise_ratio)

        if noise_nums == 0:
            logging.warning(f"Replaced {noise_nums} labels with noisy predctions.")
            return
        # Select indices to replace labels
        random.seed(self.random_seed)
        nosie_indices = random.sample(range(total_nums), noise_nums)
        
        # Choose a label from TopK predicts
        rand_topK = torch.tensor(random.choices(range(1, self.top_k), k=noise_nums))
        offset = torch.arange(0, noise_nums).long() * self.top_k

        selected_predicts = topK_predicts[nosie_indices].flatten()
        selected_predicts = (
            selected_predicts[offset + rand_topK].detach().clone().to("cpu")
        )
        datasource.trainset.targets[nosie_indices] = selected_predicts
        logging.warning(f"Replaced {noise_nums} labels with noisy predctions.")
