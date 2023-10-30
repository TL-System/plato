"""
The implementation of the moving average methods.

"""

from typing import OrderedDict as OrderedDictType

import torch


class ModelEMA:
    """A base class to compute the moving average of the model."""

    def __init__(self, beta=0.999):
        super().__init__()
        # the hyper-parameters ξ ← βξ + (1 − β)θ

        # the most important parameter beta in the moving average
        # update method.
        # as the  Exponential Moving Average (EMA) update method is the most commonly used
        # method in the self-supervised learning method. This mechanism
        # should be supported as the foundation.
        # beta:
        #   - 1: maintain the previous model without being updated with the
        #       latest weights
        #   - 0: replace the previous model with the latest weights directly.

        # With the increase of the beta, the importance of previous model will
        # also increase.
        self.beta = beta

    # def is_strictly_matched(self, src_model, dst_model):
    #     """ Whether the structure of two methods are matched strictly. """
    #     src_parameters = [for name, para in  src_model.named_parameters()
    #     dst_parameters = dst_model.named_parameters()

    def perform_average_update(self, old_weights, new_weights):
        """Perform the update average based on the old and new weights."""
        if old_weights is None:
            return new_weights
        return old_weights * self.beta + (1 - self.beta) * new_weights

    def update_model_moving_average(self, previous_model, current_model):
        """Perform the moving average to update the model.

        The input should be the model with type torch.module, thus
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

    @staticmethod
    def get_parameters_diff(parameter_a: OrderedDictType, parameter_b: OrderedDictType):
        """Get the difference between two sets of parameters"""
        # compute the divergence between encoders of local and global models
        l2_distance = 0.0
        for paraml, paramg in zip(parameter_a.items(), parameter_b.items()):
            diff = paraml[1] - paramg[1]
            # Calculate L2 norm and add to the total
            l2_distance += torch.sum(diff**2)
            print("l2_distance: ", l2_distance)

        return l2_distance.sqrt()
