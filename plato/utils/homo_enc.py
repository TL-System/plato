"""
Utility functions for homomorphric encryption with TenSEAL.
"""
import os
import pickle
import torch
import tenseal as ts
import numpy as np

from typing import OrderedDict


def get_ckks_context():
    """Obtain a TenSEAL context for encryption and decryption."""
    context_dir = ".ckks_context/"
    context_name = "context"
    try:
        with open(os.path.join(context_dir, context_name), "rb") as f:
            return ts.context_from(f.read())
    except:
        # Create a new context if not exists
        if not os.path.exists(context_dir):
            os.mkdir(context_dir)

        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        context.global_scale = 2**40

        with open(os.path.join(context_dir, context_name), "wb") as f:
            f.write(context.serialize(save_secret_key=True))
            f.close()

        return context


def encrypt_weights(
    plain_weights,
    serialize=True,
    context=None,
    encrypt_indices=None,
):
    """Flatten the model weights and encrypt the selected ones."""
    assert not context is None

    output = OrderedDict()
    # Flatten all weight tensors to a vector
    weights_vector = np.array([])
    for weight in plain_weights.values():
        weights_vector = np.append(weights_vector, weight)

    # Encrypt indices is None, full encryption
    if encrypt_indices is None:
        output["unencrypted_weights"] = None
        output["encrypt_indices"] = None
        output["encrypted_weights"] = _encrypt(weights_vector, context, True)
        return output

    # Encrypt indices is an empty list, no encryption
    if len(encrypt_indices) == 0:
        output["unencrypted_weights"] = weights_vector
        output["encrypt_indices"] = []
        output["encrypted_weights"] = None
        return output

    # Encrypt indices is a non-empty list, masked encryption
    weights_to_enc = weights_vector[encrypt_indices]
    unencrypted_weights = np.delete(weights_vector, encrypt_indices)

    # Encrypt selected weight vector
    encrypted_weights = ts.ckks_vector(context, weights_to_enc)

    # Serialize the encrypted weight vector if indicated
    if serialize:
        encrypted_weights = encrypted_weights.serialize()

    output["unencrypted_weights"] = unencrypted_weights
    output["encrypted_weights"] = encrypted_weights
    output["encrypt_indices"] = encrypt_indices

    return output


def _encrypt(data_vector, context, serialize=True):
    if serialize:
        return ts.ckks_vector(context, data_vector).serialize()
    else:
        return ts.ckks_vector(context, data_vector)


def deserialize_weights(serialized_weights, context):
    """Deserialize the encrypted weights (not decrypted yet)."""
    deserialized_weights = OrderedDict()
    for name, weight in serialized_weights.items():
        if name == "encrypted_weights" and weight != None:
            deser_weights_vector = ts.lazy_ckks_vector_from(weight)
            deser_weights_vector.link_context(context)
            deserialized_weights[name] = deser_weights_vector
        else:
            deserialized_weights[name] = weight

    return deserialized_weights


def decrypt_weights(data, weight_shapes=None, para_nums=None):
    """Decrypt the vector and restore model weights according to the shapes."""
    vector_length = []
    for para_num in para_nums.values():
        vector_length.append(para_num)

    # # Step 1: decrypt the weights
    plaintext_weights_vector = None
    encrypt_indices = data["encrypt_indices"]
    unencrypted_weights = data["unencrypted_weights"]
    if data["encrypted_weights"] != None:
        decrypted_vector = np.array(data["encrypted_weights"].decrypt())

    if encrypt_indices is None:
        # All the weights are encrypted
        plaintext_weights_vector = decrypted_vector
    elif len(encrypt_indices) == 0:
        # No weights encrypted
        plaintext_weights_vector = unencrypted_weights
    else:
        # Some encrypted and some not, put them in the right position
        vector_size = len(unencrypted_weights) + len(encrypt_indices)
        plaintext_weights_vector = np.zeros(vector_size)
        plaintext_weights_vector[encrypt_indices] = decrypted_vector

        unencrypted_indices = np.delete(range(vector_size), encrypt_indices)
        plaintext_weights_vector[unencrypted_indices] = unencrypted_weights

    # Step 2: rebuild the original weight vector by returning selected values
    decrypted_weights = OrderedDict()
    plaintext_weights_vector = np.split(
        plaintext_weights_vector, np.cumsum(vector_length)
    )[:-1]
    weight_index = 0
    for name, shape in weight_shapes.items():
        decrypted_weights[name] = plaintext_weights_vector[weight_index].reshape(shape)
        decrypted_weights[name] = torch.from_numpy(decrypted_weights[name])
        weight_index = weight_index + 1

    return decrypted_weights


def update_est(config, client_id, data):
    """Update the exposed model weights that can be estimated by adversaries."""
    unencrypted_weights = data["unencrypted_weights"]
    encrypted_indices = data["encrypt_indices"]

    model_name = config.trainer.model_name
    run_id = config.params["run_id"]
    checkpoint_path = config.params["checkpoint_path"]
    attack_prep_dir = f"{config.data.datasource}_{config.trainer.model_name}_{config.clients.encrypt_ratio}"
    if config.clients.random_mask:
        attack_prep_dir += "_random"
    if not os.path.exists(f"{checkpoint_path}/{attack_prep_dir}/"):
        os.mkdir(f"{checkpoint_path}/{attack_prep_dir}/")

    est_filename = (
        f"{checkpoint_path}/{attack_prep_dir}/{model_name}_est_{client_id}.pth"
    )
    old_est = get_est(est_filename)
    new_est = unencrypted_weights.clone().detach().double()
    if not old_est is None:
        new_est[encrypted_indices] = old_est[encrypted_indices].double()

    with open(est_filename, "wb") as est_file:
        pickle.dump(new_est, est_file)


def get_est(filename):
    """Load the estimated model, return None if not exists."""
    try:
        with open(filename, "rb") as est_file:
            return pickle.load(est_file)
    except:
        return None
