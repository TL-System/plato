"""
A factory that generates a Multi-layer perceptron (MLP), with the ability to build a fully-
connected network based on a specific configuration. This a very flexible MLP network generator to
define any type of MLP network.

Note: The general order of components in one MLP layer is:
    Schema A: From the original paper of bn and dropout.
    fc -> bn -> activation -> dropout -> ....

    Schema B: From the researcher "https://math.stackexchange.com/users/167500/pseudomarvin".
    fc -> activation -> dropout -> bn -> ....

    See more discussion on the website:
    https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout

Our work use the schema A.

Trick: One may just drop the Dropout (when you have BN) as BN eliminates the need for Dropout in
some cases, since intuitively BN provides similar regularization benefits as Dropout.

"""
from typing import Union, Dict, List
from collections import OrderedDict

from torch import nn

activations_func = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "tanh": nn.Tanh,
}


# pylint: disable=too-many-locals
def build_mlp_from_config(
    mlp_configs: Dict[str, Union[int, List[Union[str, None, dict]]]],
    layer_name_prefix: str = "layer",
):
    """
    Build the fully-connected network (Multi-layer perceptron)
    based on the input configuration.

    :param  mlp_configs: A Dict type containing the hyper-parameters for definition.
        It should contains:
            input_dim: Integar
            output_dim: Integar
            hidden_layers_dim: List[int], with length N - 1
            batch_norms: List[Union[None, dict]], with length N
            activations: List[Union[None, str]], with length N
            dropout_ratios: List[float], with length N
    :param  layer_name_prefix: A string added to the layer's name.
    """
    input_dim = mlp_configs["input_dim"]
    output_dim = mlp_configs["output_dim"]

    hidden_layers_dim = mlp_configs["hidden_layers_dim"]
    hidden_n = len(hidden_layers_dim)

    batch_norms = mlp_configs["batch_norms"]
    activations = mlp_configs["activations"]
    dropout_porbs = (
        mlp_configs["dropout_ratios"]
        if isinstance(mlp_configs["dropout_ratios"], list)
        else [mlp_configs["dropout_ratios"]]
    )

    assert len(batch_norms) == len(activations) == len(dropout_porbs)
    assert hidden_n == len(batch_norms) - 1

    def build_one_layer(
        layer_ipt_dim, layer_opt_dim, batch_norm_param, activation, dropout_prob
    ):
        """Build one layer of MLP. Default no hidden layer.

        For the structure of one MLP layer. Please access the description
        in the NOTE part.

        """
        layer_structure = OrderedDict()
        layer_structure["fc"] = nn.Linear(layer_ipt_dim, layer_opt_dim)

        if batch_norm_param is not None:
            layer_structure["bn"] = nn.BatchNorm1d(layer_opt_dim, **batch_norm_param)
        if activation is not None:
            layer_structure[activation] = activations_func[activation]()
        if dropout_prob != 0.0:
            layer_structure["drop"] = nn.Dropout(p=dropout_prob)

        return nn.Sequential(layer_structure)

    mlp_layers = OrderedDict()

    # add the final output layer to the hidden layer for building layers
    hidden_layers_dim.append(output_dim)
    for hid_id, hid_dim in enumerate(hidden_layers_dim):
        layer_input_dim = input_dim if hid_id == 0 else hidden_layers_dim[hid_id - 1]
        desired_batch_norm = batch_norms[hid_id]
        activation = activations[hid_id]
        dropout_prob = dropout_porbs[hid_id]
        built_layer = build_one_layer(
            layer_input_dim, hid_dim, desired_batch_norm, activation, dropout_prob
        )
        mlp_layers[layer_name_prefix + str(hid_id + 1)] = built_layer

    return nn.Sequential(mlp_layers)


class Model:
    """
    The Multi-layer perceptron (MLP) model.

    The implemented mlp networks are:
    - linear_mlp, The mlp with one hidden layer.
    - simclr_projection_mlp, The projection layer of SimCLR method.
    - simsiam_projection_mlp, The projection layer of SimSiam method.
    - simsiam_prediction_mlp, The prediction layer of SimSiam method.
    - byol_projection_mlp, The projection layer of BYOL method.
    - byol_prediction_mlp, The prediction layer of BYOL method.
    - moco_final_mlp, The final layer of MoCo method.
    - plato_multilayer, The Plato's multilayer.
    - customized_mlp, The customized layer.
    """

    # pylint: disable=too-few-public-methods
    @staticmethod
    def get(
        model_name: str,
        input_dim: int,
        output_dim: int,
        **kwargs: Dict[str, Union[int, List[Union[str, None, dict]]]],
    ):
        # pylint:disable=too-many-return-statements
        """Get the desired MLP model with required hyper-parameters (input_dim)."""

        if model_name == "linear_mlp":
            return build_mlp_from_config(
                mlp_configs=dict(
                    output_dim=output_dim,
                    input_dim=input_dim,
                    hidden_layers_dim=[],
                    batch_norms=[None],
                    activations=[None],
                    dropout_ratios=[0.0],
                )
            )

        if model_name == "simclr_projection_mlp":
            projection_hidden_dim = kwargs["projection_hidden_dim"]
            return build_mlp_from_config(
                dict(
                    output_dim=output_dim,
                    input_dim=input_dim,
                    hidden_layers_dim=[projection_hidden_dim],
                    batch_norms=[None, None],
                    activations=["relu", None],
                    dropout_ratios=[0.0, 0.0],
                )
            )

        if model_name == "simsiam_projection_mlp":
            projection_hidden_dim = kwargs["projection_hidden_dim"]
            return build_mlp_from_config(
                dict(
                    output_dim=output_dim,
                    input_dim=input_dim,
                    hidden_layers_dim=[
                        projection_hidden_dim,
                        projection_hidden_dim,
                    ],
                    batch_norms=[
                        dict(momentum=0.1, eps=1e-5),
                        dict(momentum=0.1, eps=1e-5),
                        dict(momentum=0.1, eps=1e-5),
                    ],
                    activations=["relu", "relu", None],
                    dropout_ratios=[0.0, 0.0, 0.0],
                )
            )

        if model_name == "simsiam_prediction_mlp":
            prediction_hidden_dim = kwargs["prediction_hidden_dim"]
            return build_mlp_from_config(
                dict(
                    output_dim=output_dim,
                    input_dim=input_dim,
                    hidden_layers_dim=[prediction_hidden_dim],
                    batch_norms=[dict(momentum=0.1, eps=1e-5), None],
                    activations=["relu", None],
                    dropout_ratios=[0.0, 0.0],
                )
            )

        if model_name == "byol_projection_mlp":
            projection_hidden_dim = kwargs["projection_hidden_dim"]
            return build_mlp_from_config(
                dict(
                    output_dim=output_dim,
                    input_dim=input_dim,
                    hidden_layers_dim=[projection_hidden_dim],
                    batch_norms=[dict(momentum=0.1, eps=1e-5), None],
                    activations=["relu", None],
                    dropout_ratios=[0.0, 0.0],
                )
            )

        if model_name == "byol_prediction_mlp":
            prediction_hidden_dim = kwargs["prediction_hidden_dim"]
            return build_mlp_from_config(
                dict(
                    output_dim=output_dim,
                    input_dim=input_dim,
                    hidden_layers_dim=[prediction_hidden_dim],
                    batch_norms=[dict(momentum=0.1, eps=1e-5), None],
                    activations=["relu", None],
                    dropout_ratios=[0.0, 0.0],
                )
            )

        if model_name == "moco_final_mlp":
            projection_hidden_dim = kwargs["projection_hidden_dim"]
            return build_mlp_from_config(
                dict(
                    output_dim=output_dim,
                    input_dim=input_dim,
                    hidden_layers_dim=[projection_hidden_dim],
                    batch_norms=[None, None],
                    activations=["relu", None],
                    dropout_ratios=[0.0, 0.0],
                )
            )

        if model_name == "plato_multilayer":
            return build_mlp_from_config(
                dict(
                    output_dim=output_dim,
                    input_dim=input_dim,
                    hidden_layers_dim=[1024, 512, 256, 128],
                    batch_norms=[None, None, None, None, None],
                    activations=["tanh", "tanh", "tanh", "tanh", None],
                    dropout_ratios=[0.0, 0.0, 0.0, 0.0, 0.0],
                )
            )

        # obtain the customized mlp laye
        # the user needs to put the corresponding hyper-parameters
        # in the 'kwargs'
        if model_name == "customized_mlp":
            return build_mlp_from_config(
                dict(output_dim=output_dim, input_dim=input_dim, **kwargs)
            )

        raise ValueError(f"No such MLP model: {model_name}")
