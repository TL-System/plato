"""
The implementation of the general Multi-layer perceptron (MLP).

I.e., build the fully-connected net based on the configs

An example of how to use this build function is presented at the bottom.

"""

from collections import OrderedDict

from torch import nn

activations_func = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax
}


def build_mlp_from_config(mlp_configs, layer_name_prefix="layer"):
    """ Build one fully-connected network based our settings.

        Note: The general order of components in one MLP layer is:
          Schema A: From the original paper of bn and dropout.
            fc -> bn -> activation -> dropout -> ....

          Schema B: From the researcher "https://math.stackexchange.com/users/167500/pseudomarvin".
            fc -> activation -> dropout -> bn -> ....
          
          See more discussion in:
           https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        
        Our work use the schema A.

        Tricks:
          - Usually, Just drop the Dropout(when you have BN)
           as BN eliminates the need for Dropout in some cases cause BN provides similar 
           regularization benefits as Dropout intuitively.


        Args:
            mlp_configs (dict):    
     """
    input_dim = mlp_configs['input_dim']
    output_dim = mlp_configs['output_dim']

    hidden_layers_dim = mlp_configs['hidden_layers_dim']
    hidden_n = len(hidden_layers_dim)

    batch_norms = mlp_configs['batch_norms']
    activations = mlp_configs['activations']
    dropout_porbs = mlp_configs['dropout_ratios'] if isinstance(
        mlp_configs['dropout_ratios'],
        list) else [mlp_configs['dropout_ratios']]

    assert len(batch_norms) == len(activations) == len(dropout_porbs)
    assert hidden_n == len(batch_norms) - 1

    def build_one_layer(layer_ipt_dim, layer_opt_dim, bn, activation,
                        dropout_prob):
        """ Build one layer of MLP. Default no hidden layer"""
        layer_structure = OrderedDict()
        layer_structure["fc"] = nn.Linear(layer_ipt_dim, layer_opt_dim)
        if bn is not None:
            default_params = dict(momentum=0.1, eps=1e-5)
            bn_params = default_params if bn == "default" else default_params.update(
                bn)
            layer_structure["bn"] = nn.BatchNorm1d(layer_opt_dim, **bn_params)
        if activation is not None:
            layer_structure[activation] = activations_func[activation](
                inplace=True)
        if dropout_prob != 0:
            layer_structure["drop"] = nn.Dropout(p=dropout_prob)

        return nn.Sequential(layer_structure)

    mlp_layers = OrderedDict()
    # add the final output layer to the hidden layer for building layers
    hidden_layers_dim.append(output_dim)
    for hid_id, hid_dim in enumerate(hidden_layers_dim):
        layer_input_dim = input_dim if hid_id == 0 else hidden_layers_dim[
            hid_id - 1]
        bn = batch_norms[hid_id]
        activation = activations[hid_id]
        dropout_prob = dropout_porbs[hid_id]
        built_layer = build_one_layer(layer_input_dim, hid_dim, bn, activation,
                                      dropout_prob)
        mlp_layers[layer_name_prefix + str(hid_id + 1)] = built_layer

    fc_net = nn.Sequential(mlp_layers)

    return fc_net


if __name__ == "__main__":
    simclr_model = dict(
        type='FullyConnectedHead',
        output_dim=256,
        input_dim=64,
        hidden_layers_dim=[64],
        batch_norms=[None, None],
        activations=["relu", None],
        dropout_ratios=[0, 0],
    )
    simclr_mlp_model = build_mlp_from_config(simclr_model)
    print(simclr_mlp_model)

    simsiam_model = dict(
        type='FullyConnectedHead',
        output_dim=2048,
        input_dim=256,
        hidden_layers_dim=[2048, 2048],
        batch_norms=["default", "default", "default"],
        activations=["relu", "relu", None],
        dropout_ratios=[0, 0, 0],
    )
    simsiam_mlp_model = build_mlp_from_config(simsiam_model)
    print(simsiam_mlp_model)

    simsiam_pred_model = dict(
        type='FullyConnectedHead',
        output_dim=2048,
        input_dim=2048,
        hidden_layers_dim=[512],
        batch_norms=["default", None],
        activations=["relu", None],
        dropout_ratios=[0, 0],
    )
    simsiam_mlp_model = build_mlp_from_config(simsiam_pred_model)
    print(simsiam_mlp_model)