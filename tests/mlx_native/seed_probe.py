"""Fresh-process logical-client replay including Torchvision stochastic data."""

import json
import sys
from pathlib import Path

import mlx.nn as nn
import numpy as np
import torch
from torchvision.transforms import RandomHorizontalFlip

from plato.algorithms.mlx_fedavg import Algorithm
from plato.callbacks.trainer import TrainerCallback
from plato.config import Config
from plato.trainers.mlx import ComposableMLXTrainer, DefaultMLXDataLoaderStrategy
from tests.mlx_native.helpers import native_config, no_device_flags


def json_tree(tree):
    if isinstance(tree, dict):
        return {key: json_tree(value) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return [json_tree(value) for value in tree]
    return tree.tolist() if hasattr(tree, "tolist") else tree


def replay(root: Path, identities: list[list[int]], master_seed: int = 29):
    import random

    recordings = {"masks": [], "order": [], "transforms": [], "losses": []}

    class Stochastic(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Dropout(0.5)
            self.linear = nn.Linear(4, 2)

        def __call__(self, x):
            masked = self.dropout(x)
            recordings["masks"].append(np.asarray(masked).tolist())
            return self.linear(masked)

    class Transformed:
        def __len__(self):
            return 9

        def __getitem__(self, index):
            image = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2) + index
            transformed = RandomHorizontalFlip()(image).flatten().numpy()
            transformed = transformed + random.random() + np.random.uniform()
            recordings["order"].append(index)
            recordings["transforms"].append(transformed.tolist())
            return transformed.astype(np.float32), index % 2

    class Losses(TrainerCallback):
        def on_train_step_end(self, trainer, config, batch=None, loss=None):
            recordings["losses"].append(float(loss.item()))

    with native_config(root, model_seed=17, training_seed=master_seed, batch_size=4):
        no_device_flags()
        trainer = ComposableMLXTrainer(
            model=Stochastic,
            callbacks=[Losses],
            data_loader_strategy=DefaultMLXDataLoaderStrategy(shuffle=True),
        )
        algorithm = Algorithm(trainer)
        baseline = algorithm.extract_weights()
        results = []
        for client_id, round_id in identities:
            trainer.set_client_id(client_id)
            trainer.current_round = round_id
            algorithm.load_weights(baseline)
            for value in recordings.values():
                value.clear()
            trainer.train_model(Config().trainer._asdict(), Transformed(), None)
            results.append(
                dict(
                    client_id=client_id,
                    round=round_id,
                    initial=json_tree(baseline),
                    weights=json_tree(algorithm.extract_weights()),
                    **{key: value[:] for key, value in recordings.items()},
                )
            )
        return results


if __name__ == "__main__":
    output = Path(sys.argv[1])
    identities = json.loads(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 29
    result = replay(output.parent / (output.stem + "-runtime"), identities, seed)
    output.write_text(json.dumps(result, sort_keys=True))
