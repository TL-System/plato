!!! note "Note"
    Your parameters in your configuration file must match the keywords in `__init__` of your model, optimizer, learning rate scheduler, or loss criterion. For example, if you want to set `base_lr` in the learning scheduler `CyclicLR`, you will need:

    ```yaml
    parameters:
        learning_rate:
            base_lr: 0.01
    ```

!!! example "model"
    All the parameter settings that need to be passed as keyword parameters when initializing the model, such as `num_classes` or `cut_layer`. The set of parameters permitted or needed depends on the model.

!!! example "optimizer"
    All the parameter settings that need to be passed as keyword parameters when initializing the optimizer, such as `lr`, `momentum`, or `weight_decay`. The set of parameters permitted or needed depends on the optimizer.

!!! example "learning_rate"
    All the parameter settings that need to be passed as keyword parameters when initializing the learning rate scheduler, such as `gamma`. The set of parameters permitted or needed depends on the learning rate scheduler.

!!! example "loss_criterion"
    All the parameter settings that need to be passed as keyword parameters when initializing the loss criterion, such as `size_average`. The set of parameters permitted or needed depends on the loss criterion.
