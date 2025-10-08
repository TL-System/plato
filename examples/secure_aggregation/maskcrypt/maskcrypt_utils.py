"""
Utility functions for MaskCrypt.
"""

import os
import pickle

import numpy as np

from plato.utils import homo_enc


def update_est(config, client_id, data):
    """Update the exposed model weights that can be estimated by adversaries."""
    unencrypted_weights, _, indices = homo_enc.extract_encrypted_model(data)
    vector_size = len(unencrypted_weights) + len(indices)
    weights_vector = np.zeros(vector_size)

    unencrypted_indices = np.delete(range(vector_size), indices)
    weights_vector[unencrypted_indices] = unencrypted_weights

    model_name = config.trainer.model_name
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
    new_est = weights_vector

    if old_est is not None:
        weights_vector[indices] = old_est[indices]

    with open(est_filename, "wb") as est_file:
        pickle.dump(new_est, est_file)


def get_est(filename):
    """Load the estimated model, return None if not exists."""
    try:
        with open(filename, "rb") as est_file:
            return pickle.load(est_file)
    except FileNotFoundError:
        return None
