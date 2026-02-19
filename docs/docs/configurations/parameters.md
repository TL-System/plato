!!! note "Note"
    Your parameters in your configuration file must match the keywords in `__init__` of your model, optimizer, learning rate scheduler, or loss criterion. For example, if you want to set `base_lr` in the learning scheduler `CyclicLR`, you will need:

    ```toml
    [parameters]
    [parameters.learning_rate]
    base_lr = 0.01
    ```

!!! example "model"
    All the parameter settings that need to be passed as keyword parameters when initializing the model, such as `num_classes` or `cut_layer`. The set of parameters permitted or needed depends on the model.

!!! example "optimizer"
    All the parameter settings that need to be passed as keyword parameters when initializing the optimizer, such as `lr`, `momentum`, or `weight_decay`. The set of parameters permitted or needed depends on the optimizer.

!!! example "learning_rate"
    All the parameter settings that need to be passed as keyword parameters when initializing the learning rate scheduler, such as `gamma`. The set of parameters permitted or needed depends on the learning rate scheduler.

!!! example "loss_criterion"
    All the parameter settings that need to be passed as keyword parameters when initializing the loss criterion, such as `size_average`. The set of parameters permitted or needed depends on the loss criterion.

## SmolVLA + LeRobot parameter contract

`Config()` keeps nested keys under `[parameters]` as dot-accessible nodes. For
the SmolVLA + LeRobot integration, define the following sections.

| Config key | Purpose | Consumption path |
| --- | --- | --- |
| `data.datasource = "LeRobot"` | Selects the robotics datasource family. | `plato.datasources.registry.get()` chooses the datasource module. |
| `trainer.type = "lerobot"` | Selects the robotics trainer backend. | `plato.trainers.registry.get()` chooses the trainer class. |
| `trainer.model_type = "smolvla"` | Selects the model family. | `plato.models.registry.get()` resolves the model factory. |
| `trainer.model_name = "smolvla"` | Selects the concrete model entry point. | `plato.models.registry.get()` resolves the model name. |
| `parameters.policy.type` | Policy family identifier (`smolvla` in v1). | Consumed by `plato/models/smolvla.py` and `plato/trainers/lerobot.py` via `Config().parameters.policy`. |
| `parameters.policy.path` | Pretrained policy source (Hub/local path). | Consumed by `plato/models/smolvla.py` via `Config().parameters.policy.path`. |
| `parameters.policy.finetune_mode` | Full fine-tune vs adapter mode switch. | Consumed by `plato/trainers/lerobot.py` to decide trainable params. |
| `parameters.policy.precision` | Runtime precision flag (`fp32`/`fp16`/`bf16`). | Consumed by `plato/trainers/lerobot.py` for dtype/autocast setup. |
| `parameters.policy.device` | Runtime device flag (`cpu`/`cuda`/`mps`). | Consumed by `plato/trainers/lerobot.py` for device placement. |
| `parameters.dataset.repo_id` | LeRobot dataset identifier. | Consumed by `plato/datasources/lerobot.py` dataset loader. |
| `parameters.dataset.delta_timestamps` | Temporal window selection per modality key. | Consumed by `plato/datasources/lerobot.py` sampling/windowing logic. |
| `parameters.transforms.*` | Image transform controls (`image_size`, `normalize`, interpolation/crop options). | Consumed by `plato/datasources/lerobot.py` preprocessing pipeline. |

### Constructor-ready dictionaries

SmolVLA/LeRobot components should convert section nodes to plain dictionaries
when passing keyword arguments into constructors:

```python
from plato.config import Config

cfg = Config()
policy_kwargs = cfg.parameters.policy._asdict()
dataset_kwargs = cfg.parameters.dataset._asdict()
transform_kwargs = cfg.parameters.transforms._asdict()
```

### Example

```toml
[data]
datasource = "LeRobot"

[trainer]
type = "lerobot"
model_type = "smolvla"
model_name = "smolvla"

[parameters.policy]
type = "smolvla"
path = "lerobot/smolvla_base"
finetune_mode = "full"
precision = "bf16"
device = "cuda"

[parameters.dataset]
repo_id = "lerobot/pusht_image"
delta_timestamps = { observation_image = [-0.2, -0.1, 0.0] }

[parameters.transforms]
image_size = [224, 224]
normalize = true
interpolation = "bilinear"
```
