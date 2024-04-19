"""
Utility functions for FedDiv
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax
import copy

def get_output(loader, net, device, criterion=None):
    net.eval()
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    net = net.to(device)
    with torch.no_grad():
        for i, (indices, (examples, labels)) in enumerate(loader):
            examples = examples.to(device)
            labels = labels.to(device)
            labels = labels.long()
            
            logits = net(examples)
            predictions = F.softmax(logits, dim=1)
            loss = criterion(predictions, labels)
            
            if i == 0:
                all_predictions = np.array(predictions.cpu())
                all_loss = np.array(loss.cpu())
                all_logits = np.array(logits.cpu())
                all_indices = np.array(indices)
            else:
                all_predictions = np.concatenate((all_predictions, predictions.cpu()), axis=0)
                all_loss = np.concatenate((all_loss, loss.cpu()), axis=0)
                all_logits = np.concatenate((all_logits, logits.cpu()), axis=0)
                all_indices = np.concatenate((all_indices, indices.cpu()), axis=0)

    return all_predictions, all_loss, all_logits, all_indices

def calculate_normalized_loss(loss):
    """
    Normalize the loss values to a range [0, 1].

    Args:
        loss (list or array): A list or numpy array of loss values to be normalized.

    Returns:
        numpy.ndarray: The normalized loss values.
    """
    # Convert the input to a numpy array if it isn't one already
    loss_array = np.array(loss)
    
    # Find the minimum loss value
    min_loss = np.min(loss_array)
    
    # Find the maximum loss value
    max_loss = np.max(loss_array)
    
    # Apply the normalization formula
    normalized_loss = (loss_array - min_loss) / (max_loss - min_loss)
    
    return normalized_loss

def local_data_splitting(loss, global_noise_filter):
    """
    This function splits local data into clean and noisy subsets based on the provided loss
    using a global noise filter model.
    
    Args:
        loss: A numpy array containing loss values of the local model's predictions.
        global_noise_filter: An object containing a pre-trained model for filtering noise.
    
    Returns:
        local_split: A dictionary containing the predicted clean and noisy flags for the samples,
                     and the estimated noise level in the local data.
    """
    
    # Calculate normalized loss, which is a preprocessing step before using the noise filter model
    normalized_loss = calculate_normalized_loss(loss)
    
    # Predict the probability of being clean for each data point using the global noise filter model
    # The model is presumably a Gaussian Mixture Model or similar, where 'means_' is an attribute
    prob_clean = global_noise_filter.predict(normalized_loss.reshape(-1, 1))
    
    # Select the probability of being clean associated with the component of the model with the lowest mean
    # This assumes that the clean data is associated with the component that has the lowest mean loss
    prob_clean = prob_clean[:, global_noise_filter.means_.argmin()]
    
    # Determine which data points are predicted clean based on the probability threshold (e.g., higher than 50%)
    pred_clean = prob_clean > 0.50
    
    # The noisy predictions are simply the complement of the clean predictions
    pred_noisy = ~pred_clean
    
    # Estimate the noise level in the dataset as the proportion of data points predicted to be noisy
    estimated_noisy_level = 1.0 - np.sum(pred_clean) / len(pred_clean)

    # Compile the results into a dictionary
    local_split = {
        'pred_clean': pred_clean,
        'pred_noisy': pred_noisy,
        'estimated_noisy_level': estimated_noisy_level,
    }
    
    # Return the dictionary containing the split information
    return local_split

def relabeling_and_reselection(local_split, sample_idx, local_output, dataset_train, de_bias, conf_threshold = 0.6,clean_set_thres = 0.1):
    # loading local data split
    pred_clean = local_split['pred_clean']
    pred_noisy = local_split['pred_noisy']
    estimated_noisy_level = local_split['estimated_noisy_level']

    y_train_given = np.array(dataset_train.targets)

    # Initialize de-biased probabilities and labels if this is the first time processing this index
    if de_bias["counter"] == 0:
        all_probs = np.max(local_output, 1)
        all_labels = np.argmax(local_output, 1)
        for id, real_sample_id in enumerate(sample_idx):
            de_bias['de_biased_probs'][real_sample_id] = all_probs[id]
            de_bias['de_biased_labels'][real_sample_id] = all_labels[id]

    # Select indices from the original sample set based on prediction cleanliness
    clean_indices = sample_idx[pred_clean]
    
    # Get pseudo-labels by finding the index of the maximum value in the model output
    pseudo_labels = np.argmax(local_output, axis=1)
    
    # Get pseudo-labels for clean samples
    clean_pseudo_labels = pseudo_labels[pred_clean]
    
    # Select the maximum output probabilities for noisy predictions
    max_prob_predictions = np.max(local_output, axis=1)

    # Noisy sample relabeling to assign pseudo-labels for noisy samples with confidence higher than threshold
    pred_noisy = pred_noisy & (max_prob_predictions > conf_threshold)
    noisy_indices = sample_idx[pred_noisy]
    noisy_pseudo_labels = pseudo_labels[pred_noisy]

    # Perform labeled sample re-selection via Predictive Consistency based Sampler (PCS)
    de_bias_clean_indices, de_bias_noisy_predictions = predictive_consistency_sampling(de_bias, noisy_indices, clean_indices, clean_pseudo_labels, noisy_pseudo_labels)
    
    # Select the indices for noisy predictions that are confident and match the de-biased labels
    relabel_indices = noisy_indices[de_bias_noisy_predictions]
    # Get new labels for confident noisy predictions
    new_labels_for_noisy = noisy_pseudo_labels[de_bias_noisy_predictions]
    
    # Update the given training labels with new labels for confident noisy predictions
    y_train_noisy_new_de_bias = np.array(y_train_given)
    y_train_noisy_new_de_bias[relabel_indices] = new_labels_for_noisy
    
    # Determine revised sample indices based on the number of times the index has been processed
    if de_bias["counter"] >= 5:
        if estimated_noisy_level > 0.20:
            revised_sample_idx = set(de_bias_clean_indices) | set(relabel_indices)
        else:
            revised_sample_idx = set(de_bias_clean_indices)
    else:
        revised_sample_idx = set(clean_indices)
    
    # Update the sample labels if estimated noise level is above the threshold (default: 0.1)
    if estimated_noisy_level <= clean_set_thres:
        revised_sample_idx = set(sample_idx)
        relabel_indices = []
    else:
        dataset_train.targets = y_train_noisy_new_de_bias
    
    # Return the revised indices, updated training dataset, and de-biasing dictionary
    return revised_sample_idx, dataset_train, de_bias, relabel_indices

def predictive_consistency_sampling(de_bias, noisy_indices, clean_indices, clean_pseudo_labels, noisy_pseudo_labels):
    # Retrieve de-biased labels for noisy and clean samples
    de_bias_labels_noisy = np.array([de_bias['de_biased_labels'][x] for x in noisy_indices])
    de_bias_labels_clean = np.array([de_bias['de_biased_labels'][x] for x in clean_indices])
    
    # Check which clean predictions match the de-biased labels
    de_bias_clean_predictions = clean_pseudo_labels == de_bias_labels_clean
    de_bias_clean_indices = clean_indices[de_bias_clean_predictions]
    
    # Determine which noisy predictions match the de-biased labels and have high confidence
    de_bias_noisy_predictions = (de_bias_labels_noisy == noisy_pseudo_labels)

    return de_bias_clean_indices, de_bias_noisy_predictions


def causal_inference(current_logit, phat, xi=0.5):
    """
    Adjust logits based on prior probabilities to perform de-biasing.

    Args:
        current_logit (array-like): The logits from the current model prediction.
        phat (array-like): The estimated prior probabilities per class.
        xi (float, optional): The scaling factor for the log prior probabilities.

    Returns:
        numpy.ndarray: The de-biased probabilities after applying softmax.
    """
    # Adjust the logits by subtracting a scaled log of the estimated prior probabilities
    adjusted_logit = current_logit - xi * np.log(phat)
    # Compute de-biased probabilities using softmax
    de_biased_prob = softmax(adjusted_logit, axis=1)

    return de_biased_prob

def update_phat(probs, phat, momentum=0.2, phat_mask=None):
    """
    Update the running estimate of the mean probability vector.

    Args:
        probs (array-like): Array of probability vectors for instances.
        phat (array-like): The current estimate of the mean probability vector.
        momentum (float): The momentum coefficient for exponential weighting.
        phat_mask (array-like, optional): An optional mask to apply to the probabilities
                                          before averaging.

    Returns:
        numpy.ndarray: The updated estimate of the mean probability vector.
    """
    # If a mask is provided, apply it to the probabilities.
    if phat_mask is not None:
        # Apply the mask and sum across the batch dimension.
        # Reshape the mask to be compatible with the probability array.
        mean_prob = (probs * phat_mask.reshape(-1, 1)).sum(axis=0) / phat_mask.sum()
    else:
        # Calculate the mean of the probability vectors across the batch dimension.
        mean_prob = probs.mean(axis=0)
    
    # Update phat with a weighted sum of the old value and the new mean probabilities.
    phat = momentum * phat + (1 - momentum) * mean_prob

    return phat

def client_cached_phat_update(de_bias, local_logits, sample_idx):
    """
    Update the cached debiased probability estimates (phat) for a client based on local model logits.
    
    Args:
        args: A configuration object containing hyperparameters.
        de_bias: A dictionary containing cached debiased probability estimates and other debiasing information.
        local_logits: The logits output by the local model for the current batch of data.
        idx: The index of the current client or data partition.
        sample_idx: The indices of the samples in the current batch.
    
    Returns:
        de_bias: The updated debiasing dictionary with new probability estimates and labels.
    """

    # Apply causal inference to obtain debiased predictions using the local logits and cached phat
    de_biased_preds = causal_inference(local_logits, de_bias['phat'])

    # Find the maximum probability from the debiased predictions for each sample
    de_biased_max_probs = np.max(de_biased_preds, axis=-1)

    de_biased_labels = np.argmax(de_biased_preds, axis=-1)
    for id, real_sample_id in enumerate(sample_idx):
        # Update the cached debiased probabilities for the given sample indices
        de_bias['de_biased_probs'][real_sample_id] = de_biased_max_probs[id]
        # Update the cached debiased labels for the given sample indices
        de_bias['de_biased_labels'][real_sample_id] = de_biased_labels[id]

    
    # de_bias['de_biased_probs'][sample_idx] = de_biased_max_probs
    # de_bias['de_biased_labels'][sample_idx] = np.argmax(de_biased_preds, axis=-1)

    # Create a mask for samples where the maximum debiased probability is greater than a confidence threshold (0.85)
    prob_mask = de_biased_max_probs > 0.85

    # If there are samples above the confidence threshold, update phat for the given client or data partition
    if prob_mask.sum() > 0:
        # Update phat using a moving average with momentum; only use logits from confident samples
        de_bias['phat'] = update_phat(softmax(local_logits, axis=-1)[prob_mask], de_bias['phat'])

    # Return the updated debiasing information
    return de_bias
