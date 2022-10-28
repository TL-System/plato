"""
Utility functions for homomorphric encryption with TenSEAL.
"""
import os
import pickle
import zlib
from typing import OrderedDict

import numpy as np
import tenseal as ts
import torch


def get_ckks_context():
    """Obtain a TenSEAL context for encryption and decryption."""
    context_dir = ".ckks_context/"
    context_name = "context"
    try:
        with open(os.path.join(context_dir, context_name), "rb") as f:
            return ts.context_from(f.read())
    except:
        # Create a new context if it does not exist
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
    indices=None,
):
    """Flatten the model weights and encrypt the selected ones."""
    assert not context is None

    # Step 1: flatten all weight tensors to a vector
    weights_vector = np.array([])
    for weight in plain_weights.values():
        weights_vector = np.append(weights_vector, weight)

    # Step 2: set up the indices for encrypted weights
    encrypt_indices = None
    if indices is None:
        encrypt_indices = np.arange(len(weights_vector)).tolist()
    else:
        encrypt_indices = indices
    encrypt_indices.sort()

    # Step 3: separate weights into encrypted and unencrypted ones
    unencrypted_weights = np.delete(weights_vector, encrypt_indices)
    weights_to_enc = weights_vector[encrypt_indices]

    if len(weights_to_enc) == 0:
        encrypted_weights = None
    else:
        encrypted_weights = _encrypt(weights_to_enc, context, serialize)

    # Finish by wrapping up the information
    output = wrap_encrypted_model(
        unencrypted_weights, encrypted_weights, encrypt_indices
    )

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
        if name == "encrypted_weights" and weight is not None:
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

    # Step 1: decrypt the encrypted weights
    plaintext_weights_vector = None
    unencrypted_weights, encrypted_weights, indices = extract_encrypted_model(data)

    if len(indices) != 0:
        decrypted_vector = np.array(encrypted_weights.decrypt())

        vector_size = len(unencrypted_weights) + len(indices)
        plaintext_weights_vector = np.zeros(vector_size)
        plaintext_weights_vector[indices] = decrypted_vector

        unencrypted_indices = np.delete(range(vector_size), indices)
        plaintext_weights_vector[unencrypted_indices] = unencrypted_weights
    else:
        plaintext_weights_vector = unencrypted_weights

    # Step 2: rebuild the original weight vector
    decrypted_weights = OrderedDict()
    plaintext_weights_vector = np.split(
        plaintext_weights_vector, np.cumsum(vector_length)
    )[:-1]
    weight_index = 0
    for name, shape in weight_shapes.items():
        decrypted_weights[name] = plaintext_weights_vector[weight_index].reshape(shape)
        try:
            decrypted_weights[name] = torch.from_numpy(decrypted_weights[name])
        except:
            # PyTorch does not exist, just return numpy array and handle it somewhere else.
            decrypted_weights[name] = decrypted_weights[name]
        weight_index = weight_index + 1

    return decrypted_weights


def wrap_encrypted_model(unencrypted_weights, encrypted_weights, indices):
    """Wrap up the encrypted model in a dict as the message between server and client."""
    message = {
        "unencrypted_weights": unencrypted_weights,
        "encrypted_weights": encrypted_weights,
        "indices": indices,
    }

    return message


def extract_encrypted_model(data):
    """Extract infromation from the message of encrytped model"""
    unencrypted_weights = data["unencrypted_weights"]
    encrypted_weights = data["encrypted_weights"]
    indices = data["indices"]
    return unencrypted_weights, encrypted_weights, indices


def indices_to_bitmap(indices):
    """Turn a list of indices into a bitmap."""
    if indices == []:
        # In case of empty list
        return indices
    bit_array = np.zeros(np.max(indices) + 1, dtype=np.int8)
    bit_array[indices] = 1
    bitmap = np.packbits(bit_array)

    # Compress the bitmap before sending it out
    compressed_bitmap = zlib.compress(pickle.dumps(bitmap))
    return compressed_bitmap


def bitmap_to_indices(bitmap):
    """Translate a bitmap back to a list of indices."""
    if bitmap == []:
        # In case of empty list
        return bitmap

    decompressed_bitmap = pickle.loads(zlib.decompress(bitmap))
    bit_array = np.unpackbits(decompressed_bitmap)
    indices = np.where(bit_array == 1)[0].tolist()
    return indices
