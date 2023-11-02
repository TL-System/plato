"""
The implementation of the moving average methods.
"""


class ModelEMA:
    """A base class to compute the moving average of the model."""

    def __init__(self, beta=0.999):
        super().__init__()
        # The hyper-parameters ξ ← βξ + (1 − β)θ

        # The most important parameter beta in the moving average
        #   update method.
        # beta:
        #   - 1: maintain the previous model without being updated with the
        #       latest weights
        #   - 0: replace the previous model with the latest weights directly.

        # With the increase of the beta, the importance of previous model will increase.
        self.beta = beta

    def perform_average_update(self, old_weights, new_weights):
        """Perform the update average based on the old and new weights."""
        if old_weights is None:
            return new_weights
        return old_weights * self.beta + (1 - self.beta) * new_weights

    def update_model_moving_average(self, previous_model, current_model):
        """Perform the moving average to update the model.

        The input should be the model with type torch.layer, thus
        its parameter can be obtained by model.parameters() ->
        OrderDict
        """
        for previous_params, current_params in zip(
            previous_model.parameters(), current_model.parameters()
        ):
            old_weight, up_weight = previous_params.data, current_params.data
            previous_params.data = self.perform_average_update(old_weight, up_weight)

    def update_parameters_moving_average(
        self, previous_parameters: dict, current_parameters: dict
    ):
        """Perform the moving average to update the model.

        The weights is directly a OrderDict containing the
        parameters that will be assigned to the model by using moving
        average.
        """
        for parameter_name in previous_parameters:
            old_weight = previous_parameters[parameter_name]
            cur_weight = current_parameters[parameter_name]
            current_parameters[parameter_name] = self.perform_average_update(
                old_weight, cur_weight
            )

        return current_parameters
