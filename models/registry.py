"""
Having a registry of all available classes is convenient for retrieving an instance
based on a configuration at run-time.
"""

from models import mnist_cnn, fashion_cnn, cifar_resnet, cifar_wideresnet, cifar_vgg

registered_models = [
    mnist_cnn.Model, fashion_cnn.Model, cifar_resnet.Model, cifar_wideresnet.Model, cifar_vgg.Model
]


def get(model_name):
    """Get the model with the provided name."""
    model = None
    for registered_model in registered_models:
        if registered_model.is_valid_model_name(model_name):
            model = registered_model.get_model_from_name(model_name)
            break

    if model is None:
        raise ValueError('No such model: {}'.format(model_name))

    return model
