"""
The implementation of the moving average methods.

"""


class ModelEMA:

    def __init__(self, beta):
        super().__init__()
        # the hyper-parameters ξ ← βξ + (1 − β)θ
        self.beta = beta

    # def is_strictly_matched(self, src_model, dst_model):
    #     """ Whether the structure of two methods are matched strictly. """
    #     src_parameters = [for name, para in  src_model.named_parameters()
    #     dst_parameters = dst_model.named_parameters()

    def perform_average_update(self, old_weights, new_weights):
        """ Perform the update average based on the old and new weights. """
        if old_weights is None:
            return new_weights
        return old_weights * self.beta + (1 - self.beta) * new_weights

    def update_model_moving_average(self, previous_model, current_model):
        """ Perform the moving average to update the model. """
        for current_params, previous_params in zip(
                current_model.parameters(), previous_model.parameters()):
            up_weight, old_weight = current_params.data, previous_params.data
            previous_params.data = self.perform_average_update(
                old_weight, up_weight)
