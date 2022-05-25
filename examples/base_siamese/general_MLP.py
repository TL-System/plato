"""
The implementation of the general Multi-layer perceptron (MLP).

I.e., build the fully-connected net based on the configs

An example of how to use this build function is presented at the bottom.

"""

from collections import OrderedDict

from torch import nn


def build_mlp_from_config(mlp_configs):
    """ Build one fully-connected network based our settings.

        Args:
            mlp_configs (dict):    
     """

    out_classes_n = mlp_configs['num_classes']

    hidden_layer_dims = mlp_configs['hidden_layer_size']
    hidden_n = len(hidden_layer_dims)

    drop_out_porbs = mlp_configs['dropout_ratio'] if isinstance(
        mlp_configs['dropout_ratio'],
        list) else [mlp_configs['dropout_ratio']]
    drop_out_porbs = drop_out_porbs * hidden_n if len(
        drop_out_porbs) == 1 else drop_out_porbs

    fc_strcuture = OrderedDict()

    # the first layer
    fc_strcuture['fc1'] = nn.Linear(mlp_configs['in_channels'],
                                    hidden_layer_dims[0])
    fc_strcuture['relu1'] = nn.ReLU()
    fc_strcuture['drop1'] = nn.Dropout(p=drop_out_porbs[0])
    # the hidden layer
    for hidden_l_i, layer_dim in enumerate(hidden_layer_dims):
        layer_dim = hidden_layer_dims[hidden_l_i]
        if hidden_l_i == hidden_n - 1:  # the final prediction layer
            layer_name = 'fcf'  # the final layer

            fc_strcuture[layer_name] = nn.Linear(layer_dim, out_classes_n)
            fc_strcuture['sigmoid'] = nn.Sigmoid()

        else:
            next_layer_in_dim = hidden_layer_dims[hidden_l_i + 1]
            layer_name = "fc" + str(hidden_l_i + 2)
            relu_name = "relu" + str(hidden_l_i + 2)
            dropout_name = "dropout" + str(hidden_l_i + 2)
            fc_strcuture[layer_name] = nn.Linear(layer_dim, next_layer_in_dim)
            fc_strcuture[relu_name] = nn.ReLU()
            fc_strcuture[dropout_name] = nn.Dropout(
                p=drop_out_porbs[hidden_l_i + 1])

    fc_net = nn.Sequential(fc_strcuture)

    return fc_net


if __name__ == "__main__":
    fuse_model = dict(
        type='FullyConnectedHead',
        num_classes=400,
        in_channels=512 * 3,
        hidden_layer_size=[1024, 512],
        dropout_ratio=0.5,
    )
    fc_model = build_mlp_from_config(fuse_model)
    print(fc_model)
