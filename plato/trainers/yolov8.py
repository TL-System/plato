"""The YOLOV8 model for PyTorch."""
import logging
import os

import torch

from ultralytics.data.build import (
    InfiniteDataLoader,
    seed_worker,
)

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer

from plato.config import Config
from plato.trainers import basic


class SampledDetectionTrainer(DetectionTrainer):
    """Inheriting from DetectionTrainer to support a customized datasource and sampler."""

    def __init__(
        self, train_loader, test_loader, sampler, cfg=DEFAULT_CFG, overrides=None
    ):
        super().__init__(cfg=cfg, overrides=overrides)
        self.trainloader = train_loader
        self.testloader = test_loader
        self.sampler = sampler

    def get_dataloader(self, dataset_path, batch_size, mode="train", rank=0):
        if mode == "train":
            return self.trainloader
        else:
            return self.testloader


class Trainer(basic.Trainer):
    """The YOLOV8 trainer."""

    def __init__(self, model=None, callbacks=None):
        super().__init__(callbacks=callbacks)

        self.overrides = dict(
            model=Config().parameters.model.type,
            data=Config().data.data_params,
            batch=Config().trainer.batch_size,
            epochs=Config().trainer.epochs,
            device=str(self.client_id % 4)
            if Config().device() == "cuda"
            else Config().device(),
            workers=1,
            seed=Config().clients.random_seed,
        )

        print(self.overrides)

    def create_dataloader(self, cfg, dataset, batch_size, sampler=None, mode="train"):
        """Creates a PyTorch dataloader for YOLOv8 datasets."""
        assert mode in ["train", "val"]
        shuffle = mode == "train"

        num_devices = torch.cuda.device_count()  # number of CUDA devices
        workers = cfg["workers"] if mode == "train" else cfg["workers"] * 2
        num_workers = min(
            [
                os.cpu_count() // max(num_devices, 1),
                batch_size if batch_size > 1 else 0,
                workers,
            ]
        )  # number of workers
        loader = InfiniteDataLoader
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + cfg["seed"])
        return loader(
            dataset=dataset,
            batch_size=Config().trainer.batch_size
            if batch_size is None
            else batch_size,
            shuffle=shuffle and sampler is None,
            num_workers=num_workers,
            sampler=sampler,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
        )

    # pylint: disable=unused-argument
    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        """The custom train loader for YOLOv8."""
        return self.create_dataloader(
            cfg=self.overrides,
            dataset=trainset,
            batch_size=batch_size,
            sampler=sampler,
            mode="train",
        )

    # pylint: disable=unused-argument
    def get_test_loader(self, batch_size, dataset, sampler):
        """The custom test loader for YOLOv8."""
        return self.create_dataloader(
            cfg=self.overrides,
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            mode="val",
        )

    # pylint: disable=unused-argument
    def train_model(self, config, trainset, sampler, **kwargs):
        """The training loop for YOLOv8.

        Arguments:
        config: A dictionary of configuration parameters.
        trainset: The training dataset.
        """
        trainer = SampledDetectionTrainer(
            train_loader=self.get_train_loader(config["batch_size"], trainset, sampler),
            test_loader=self.get_test_loader(config["batch_size"], trainset, sampler),
            sampler=self.sampler,
            overrides=self.overrides,
        )

        trainer.model = trainer.get_model(
            weights=self.model, cfg=Config().parameters.model.cfg
        )
        trainer.train()
        self.model = trainer.model

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)

    def test_model(self, config, testset, sampler=None, **kwargs):
        """The test loop for YOLOv8.

        Arguments:
        config: A dictionary of configuration parameters.
        testset: The test dataset.
        """
        # Testing the updated model directly at the server
        logging.info("[%s] Started model testing.", self)

        trainer = SampledDetectionTrainer(
            train_loader=self.get_train_loader(config["batch_size"], testset, sampler),
            test_loader=self.get_test_loader(config["batch_size"], testset, sampler),
            sampler=self.sampler,
            overrides=self.overrides,
        )

        trainer.test_loader = trainer.testloader
        trainer.model = trainer.get_model(
            weights=self.model, cfg=Config().parameters.model.cfg
        )

        validator = trainer.get_validator()
        # "metrics/precision(B)", "metrics/recall(B)",
        # "metrics/mAP50(B)", "metrics/mAP50-95(B)"
        stats = validator(trainer=None, model=self.model)

        map50 = stats["metrics/mAP50(B)"]
        validator.print_results()

        return map50
