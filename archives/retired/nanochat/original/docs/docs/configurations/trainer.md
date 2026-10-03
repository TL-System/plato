!!! example "type"
    The type of the trainer. The following types are available:

    - `basic` a basic trainer with a standard training loop.
    - `composable` the strategy-based trainer that exposes loss, optimiser, scheduler, data-loader, model-update, and testing strategies directly.
    - `timm_basic` a basic trainer with the [timm](https://timm.fast.ai/) learning rate scheduler.
    - `diff_privacy` a trainer that supports local differential privacy in its training loop by adding noise to the gradients during each step of training.
    - `HuggingFace` a trainer for Hugging Face causal language models and tokenizers.
    - `nanochat` a trainer for Nanochat language-model workloads.
    - `lerobot` a trainer for LeRobot / SmolVLA workloads.
    - `split_learning` a trainer that supports the split learning framework.
    - `self_supervised_learning` a trainer that supports personalized federated learning based on self supervised learning.
    - `gan` a trainer for Generative Adversarial Networks (GANs).
    - `pfedgraph` a trainer used by the pFedGraph personalized federated learning algorithm.

    !!! note "Framework shortcut"
        Plato also supports `framework = "mlx"`, which resolves to the MLX trainer backend.

    !!! example "max_physical_batch_size"
        The limit on the physical batch size when using the `diff_privacy` trainer.

        Default value: `128`. The GPU memory usage of one process training the ResNet-18 model is around 2817 MB.

    !!! example "dp_epsilon"
        Total privacy budget of epsilon with the `diff_privacy` trainer.

        Default value: `10.0`

    !!! example "dp_delta"
        Total privacy budget of delta with the `diff_privacy` trainer.

        Default value: `1e-5`

    !!! example "dp_max_grad_norm"
        The maximum norm of the per-sample gradients with the `diff_privacy` trainer. Any gradient with norm higher than this will be clipped to this value.

        Default value: `1.0`

!!! example "rounds"
    The maximum number of training rounds.

    `round` could be any positive integer.

!!! example "max_concurrency"
    The maximum number of clients (of each edge server in cross-silo training) running concurrently on each available GPU. If this is not defined, no new processes are spawned for training.

    !!! note "Note"
        Plato will automatically use all available GPUs to maximize the concurrency of training, launching the same number of clients on every GPU. If `max_concurrency` is 7 and 3 GPUs are available, 21 client processes will be launched for concurrent training.

!!! example "target_accuracy"
    The target accuracy of the global model.

!!! example "target_perplexity"
    The target perplexity of the global Natural Language Processing (NLP) model.

!!! example "epochs"
    The total number of epochs in local training in each communication round.

!!! example "batch_size"
    The size of the mini-batch of data in each step (iteration) of the training loop.

!!! example "gradient_accumulation_steps"
    The number of mini-batches to accumulate before applying an optimizer step.

    This is commonly used by the `HuggingFace` trainer to keep memory usage manageable when fine-tuning larger language models.

!!! example "gradient_checkpointing"
    Whether activation checkpointing should be enabled when supported by the trainer/model stack.

    This is especially useful for Hugging Face LLM fine-tuning.

!!! example "bf16"
    Whether bfloat16 should be used when supported by the runtime.

!!! example "fp16"
    Whether float16 should be used when supported by the runtime.

!!! example "optimizer"
    The type of the optimizer. The following options are supported:

    - `Adam`
    - `Adadelta`
    - `Adagrad`
    - `AdaHessian` (from the `torch_optimizer` package)
    - `AdamW`
    - `SparseAdam`
    - `Adamax`
    - `ASGD`
    - `LBFGS`
    - `NAdam`
    - `RAdam`
    - `RMSprop`
    - `Rprop`
    - `SGD`

!!! example "lr_scheduler"
    The learning rate scheduler. The following learning rate schedulers are supported:

    - `CosineAnnealingLR`
    - `LambdaLR`
    - `MultiStepLR`
    - `StepLR`
    - `ReduceLROnPlateau`
    - `ConstantLR`
    - `LinearLR`
    - `ExponentialLR`
    - `CyclicLR`
    - `CosineAnnealingWarmRestarts`

    Alternatively, all four schedulers from [timm](https://timm.fast.ai/schedulers) are supported if `lr_scheduler` is specified as `timm` and `trainer -> type` is specified as `timm_basic`. For example, to use the `SGDR` scheduler, we specify `cosine` as `sched` in its arguments (`parameters -> learning_rate`):

    ```toml
    [trainer]
    type = "timm_basic"

    [parameters]

    [parameters.learning_rate]
    sched = cosine
    min_lr = 1.e-6
    warmup_lr = 0.0001
    warmup_epochs = 3
    cooldown_epochs = 10
    ```

!!! example "loss_criterion"
    The loss criterion. The following options are supported:

    - `L1Loss`
    - `MSELoss`
    - `BCELoss`
    - `BCEWithLogitsLoss`
    - `NLLLoss`
    - `PoissonNLLLoss`
    - `CrossEntropyLoss`
    - `HingeEmbeddingLoss`
    - `MarginRankingLoss`
    - `TripletMarginLoss`
    - `KLDivLoss`
    - `NegativeCosineSimilarity`
    - `NTXentLoss`
    - `BarlowTwinsLoss`
    - `DCLLoss`
    - `DCLWLoss`
    - `DINOLoss`
    - `PMSNCustomLoss`
    - `PMSNLoss`
    - `SwaVLoss`
    - `SymNegCosineSimilarityLoss`
    - `TiCoLoss`
    - `VICRegLoss`
    - `VICRegLLoss`
    - `MSNLoss`

    !!! note "Optional dependency"
        Self-supervised loss criteria are loaded lazily and require the optional `lightly` package in the runtime environment.

!!! example "global_lr_scheduler"
    Whether the learning rate should be scheduled globally (`true`) or not (`false`).
    If `true`, the learning rate of the first epoch in the next communication round is scheduled based on that of the last epoch in the previous communication round.

!!! example "model_type"
    The repository where the machine learning model should be retrieved from. The following options are available:

    - `cnn_encoder` (for generating various encoders by extracting from CNN models such as ResNet models)
    - `general_multilayer` (for generating a multi-layer perceptron using a provided configuration)
    - `huggingface` (for [HuggingFace](https://huggingface.co/models) causal language models)
    - `torch_hub` (for models from [PyTorch Hub](https://pytorch.org/hub/))
    - `vit` (for Vision Transformer models from [HuggingFace](https://huggingface.co/models), [Tokens-to-Token ViT](https://github.com/yitu-opensource/T2T-ViT), and [Deep Vision Transformer](https://github.com/zhoudaquan/dvit_repo))
    - `smolvla` (for LeRobot / SmolVLA robotics policies)

    The name of the model should be specified below, in `model_name`.

    !!! note "Note"
        For `vit`, please replace the `/` in model name from [https://huggingface.co/models](https://huggingface.co/models) with `@`. For example, use `google@vit-base-patch16-224-in21k` instead of `google/vit-base-patch16-224-in21k`. If you do not want to use the pretrained weights, set `parameters -> model -> pretrained` to `false`, as in the following example:

        ```toml
        [parameters]
        [parameters.model]
        pretrained = false
        ```

!!! example "model_name"
    The name of the machine learning model. The following options are available:

    - `lenet5`
    - `resnet_x`
    - `vgg_x`
    - `dcgan`
    - `multilayer`
    - `nanochat`
    - `smolvla`

    !!! note "Note"
        If the `model_type` above specified a model repository, supply the name of the model, such as `gpt2`, `HuggingFaceTB/SmolLM2-135M`, or `smolvla`, here.

        For `resnet_x`, x = 18, 34, 50, 101, or 152; for `vgg_x`, x = 11, 13, 16, or 19.

!!! example "tokenizer_name"
    An optional tokenizer identifier to use instead of `trainer.model_name`.

    This is mainly useful for Hugging Face language-model workloads where the tokenizer/chat template comes from a separate repository.
