import torch
import logging
from plato.config import Config
from scipy.stats import norm
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
import pickle
import os
from sklearn.decomposition import PCA
import torch.nn.functional as F
from collections import OrderedDict


def get():

    detector_type = (
        Config().server.detector_type
        if hasattr(Config().server, "detector_type")
        else None
    )

    if detector_type is None:
        logging.info("No defence is applied.")
        return lambda x: x

    if detector_type in registered_detectors:
        registered_defence = registered_detectors[detector_type]
        logging.info(f"Clients perform {detector_type} attack.")
        return registered_defence

    raise ValueError(f"No such defence: {detector_type}")


def flatten_weights(weights):
    flattened_weights = []

    for weight in weights:
        flattened_weight = []
        for name in weight.keys():
            flattened_weight = (
                weight[name].view(-1)
                if not len(flattened_weight)
                else torch.cat((flattened_weight, weight[name].view(-1)))
            )

        flattened_weights = (
            flattened_weight[None, :]
            if not len(flattened_weights)
            else torch.cat((flattened_weights, flattened_weight[None, :]), 0)
        )
    return flattened_weights


def flatten_weight(weight):

    flattened_weight = []
    for name in weight.keys():
        flattened_weight = (
            weight[name].view(-1)
            if not len(flattened_weight)
            else torch.cat((flattened_weight, weight[name].view(-1)))
        )
    return flattened_weight


def lbfgs(weights_attacked, global_weights_record, gradients_record, last_weights):
    """Approximate integrated Hessian value"""
    # Transfer lists of tensor into tensor matrix
    global_weights = torch.stack(global_weights_record)
    gradients = torch.stack(gradients_record)

    global_times_gradients = torch.matmul(global_weights, gradients.T)
    global_times_global = torch.matmul(global_weights, global_weights.T)

    # Get its diagonal matrix and lower triangular submatrix
    R_k = np.triu(global_times_gradients.numpy())
    L_k = global_times_gradients - torch.tensor(R_k)
    # Step 3 in Algorithm 1
    sigma_k = torch.matmul(
        torch.transpose(global_weights_record[-1], 0, -1), gradients_record[-1]
    ) / (
        torch.matmul(torch.transpose(gradients_record[-1], 0, -1), gradients_record[-1])
    )
    D_k_diag = torch.diag(global_times_gradients)

    # upper_mat = (sigma_k * global_times_global)
    upper_mat = torch.cat(((sigma_k * global_times_global), L_k), dim=1)
    lower_mat = torch.cat((L_k.T, -torch.diag(D_k_diag)), dim=1)
    mat = torch.cat((upper_mat, lower_mat), dim=0)
    mat_inv = torch.inverse(mat)

    v = (
        torch.mean(weights_attacked) - last_weights
    )  # deltas_attacked from selected clients

    approx_prod = sigma_k * v
    p_mat = torch.cat(
        (torch.matmul(global_weights, sigma_k * v), torch.matmul(gradients, v)), dim=0
    )
    approx_prod -= torch.matmul(
        torch.matmul(
            torch.cat((sigma_k * global_weights.T, gradients.T), dim=1), mat_inv
        ),
        p_mat,
    )

    return approx_prod


def gap_statistics(score):
    logging.info(f"in gap function")
    nrefs = 10
    ks = range(1, 3)
    gaps = np.zeros(len(ks))
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks))
    min = np.min(score)
    max = np.max(score) + 1  #!
    score = (score - min) / (max - min)
    for i, k in enumerate(ks):
        estimator = KMeans(n_clusters=k)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum(
            [np.square(score[m] - center[label_pred[m]]) for m in range(len(score))]
        )
        logging.info(f"line 128")
        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum(
                [np.square(rand[m] - center[label_pred[m]]) for m in range(len(rand))]
            )
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
    logging.info(f"line 105")

    select_k = None
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i + 1
            break
    if select_k == 1:
        print("No attack detected!")
        return 0
    else:
        print("Attack Detected!")
        return 1


def detection(score):
    logging.info(f"in detection function")
    estimator = KMeans(n_clusters=2)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    if np.mean(score[label_pred == 0]) < np.mean(score[label_pred == 1]):
        # cluster with smaller mean value is clean clients
        clean_ids = np.where(label_pred == 0)[0]
        malicious_ids = np.where(label_pred == 1)[0]
    else:
        clean_ids = np.where(label_pred == 1)[0]
        malicious_ids = np.where(label_pred == 0)[0]
    logging.info(f"clean_ids: %s", clean_ids)
    return malicious_ids, clean_ids


def fl_detector(baseline_weights, weights_attacked, deltas_attacked):
    """https://arxiv.org/pdf/2207.09209.pdf"""
    clean_weights = weights_attacked

    flattened_weights_attacked = flatten_weights(weights_attacked)
    flattened_baseline_weights = flatten_weight(baseline_weights)
    flattened_deltas_attacked = flatten_weights(deltas_attacked)

    window_size = 3
    malicious_ids = []

    file_path = "./records.pkl"
    if os.path.exists(file_path):
        logging.info(f"fldetector loading parameters from file.")
        with open(file_path, "rb") as file:
            global_weights_record = pickle.load(file)
            gradients_record = pickle.load(file)
            last_weights = pickle.load(file)
            last_gradients = pickle.load(file)
            malicious_score = pickle.load(file)

        if len(global_weights_record) >= window_size + 1:
            # Make predication by Cauchy mean value theorem
            hvp = lbfgs(
                flattened_weights_attacked,
                global_weights_record,
                gradients_record,
                last_weights,
            )

            # it assumes all clients get selected at each round.
            pred_grad = torch.add(last_gradients, hvp)
            logging.info(f"shape of stack(pred_grad): %s", pred_grad.shape)

            # Calculate distance for scoring
            distance = torch.norm(
                (pred_grad - flattened_deltas_attacked), dim=1
            ).numpy()
            logging.info(f"the distance is: %s", distance)
            # Normalize distance
            distance = distance / np.sum(distance)
            logging.info(f"the distance after normalization: %s", distance)
            # add new distance score into malicious score record
            # malicious_score = np.row_stack((malicious_score, distance))
            # Prepare for moving averaging
            malicious_score = distance  # np.row_stack((malicious_score, distance))
            logging.info(f"the malicious score: %s", malicious_score)
            # clustering
            # if malicious_score.shape[0] >= 1:
            logging.info(f"line 176 for checkpoint")

            if gap_statistics(
                malicious_score
            ):  # np.sum(malicious_score[-10:], axis=0)):
                logging.info(f"malicious clients detected!")
                malicious_ids, clean_ids = detection(
                    malicious_score
                )  # np.sum(malicious_score[-10:], axis=0))

            # remove poisoned weights
            clean_weights = []
            for i, weight in enumerate(weights_attacked):
                if i not in malicious_ids:
                    clean_weights.append(weight)

    else:
        logging.info(f"initializing fl parameter record")
        global_weights_record = []
        gradients_record = []
        last_gradients = torch.zeros(len(flattened_baseline_weights))
        last_weights = torch.zeros(len(flattened_baseline_weights))
        malicious_score = []

    # update record
    logging.info(f"line 183: updating record")
    global_weights_record.append(flattened_baseline_weights - last_weights)
    logging.info(f"len %d", len(global_weights_record))
    gradients_record.append(
        torch.mean(flattened_deltas_attacked, dim=0) - last_gradients
    )
    last_weights = flattened_baseline_weights
    last_gradients = torch.mean(flattened_deltas_attacked, dim=0)

    # save into local file
    file_path = "./records.pkl"
    with open(file_path, "wb") as file:
        pickle.dump(global_weights_record, file)
        pickle.dump(gradients_record, file)
        pickle.dump(last_weights, file)
        pickle.dump(last_gradients, file)
        pickle.dump(malicious_score, file)
    logging.info(f"malicious_ids: %s", malicious_ids)
    return malicious_ids, clean_weights


def encoder_decoder(weights_attacked):

    # load pre-trained encoder and decoder model
    # apply it to weights attacked
    # obtain reconstruction error for all weights attacked
    return reconstruction_errors


def spectral_anomaly_detection(baseline_weights, weights_attacked, deltas_attacked):
    """https://arxiv.org/pdf/2002.00211.pdf"""
    flattened_weights = flatten_weights(weights_attacked)

    reconstruction_errors = encoder_decoder(flattened_weights)

    threshold = np.mean(reconstruction_errors)

    malicious_ids = np.where(reconstruction_errors > threshold)

    clean_weights = []
    for i, weight in enumerate(weights_attacked):
        if i not in malicious_ids:
            clean_weights.append(weight)

    return malicious_ids, clean_weights


def mab_rfl(baseline_weights, weights_attacked, deltas_attacked):
    alpha = 0.9
    # flatten weights
    flattened_weights = flatten_weights(weights_attacked)

    # load history for momentum avg
    file_path = "./mab_rfl_records.pkl"
    if os.path.exists(file_path):
        logging.info(f"mab-rfl loading parameters from file.")
        with open(file_path, "rb") as file:
            last_weights = pickle.load(file)
    else:
        # Initilization
        last_weights = torch.zeros(len(flattened_weights))

    logging.info(f"start calculating momentum and normalization...")
    # calculate momentum avg and normalization
    weights_mom = []
    weights_norm = []
    for new_weight, last_weight in zip(flattened_weights, last_weights):
        weights_temp = new_weight + alpha * last_weight
        weights_mom.append(weights_temp)
        weights_norm.append(weights_temp / torch.linalg.norm(weights_temp))
    logging.info(f"weights_norm: %s", weights_norm)
    logging.info(f"Finished calculating momentum and normalization...")
    # update history
    file_path = "./records.pkl"
    with open(file_path, "wb") as file:
        pickle.dump(weights_mom, file)

    # Apply PCA
    """
    weights_for_pca = []
    for j, weight in enumerate(weights_attacked):
        start_index = 0
        weight_temp = OrderedDict()
        logging.info(f"j: %d", j)
        for name, weight in weight.items():
            weight_norm = weights_norm[j]
            #logging.info(f"weight_norm: %s ", weight_norm[start_index : start_index + len(weight.view(-1))])
            weight_temp[name] = weight_norm[start_index : start_index + len(weight.view(-1))]#.reshape(weight.shape)
            start_index += len(weight.view(-1))

        weights_for_pca.append(weight_temp)

    key_weights = []
    for weight in weights_for_pca:
        pca = PCA(n_components=3)  # adjust as needed
        logging.info(f"np.array(weight.values()): %s", np.array(weight.values()))
        key_weights.append(pca.fit_transform(np.array(weight.values())))
    """
    # pca = PCA(n_components=3)  # adjust as needed
    # logging.info(f"np.array(weight.values()): %s", np.array(weight.values()))
    # key_weights.append(pca.fit_transform(np.array(weight.values())))
    logging.info(f"applying PCA.")
    # logging.info(f"key_weights are %s",key_weights)
    # applying agglomerative clustering algorithm
    clustering = AgglomerativeClustering(n_clusters=2)
    weights_norm = torch.stack(weights_norm)
    clustering.fit(weights_norm)

    # Calculate the mean of each cluster
    cluster_points1 = weights_norm[clustering.labels_ == 0]
    cluster_mean1 = cluster_points1.mean(dim=0)
    cluster_points2 = weights_norm[clustering.labels_ == 1]
    cluster_mean2 = cluster_points2.mean(dim=0)

    # calculate cosine similarity and compare to a threshold alpha; and return malicious ones
    if F.cosine_similarity(cluster_mean1, cluster_mean2, dim=0) < alpha:
        logging.info(f"No malicious clients detected.")
        return [], weights_attacked
    else:
        if cluster_points1.dim() < cluster_points2.dim():
            malicious_ids = np.where(clustering.labels_ == 0)
        else:
            malicious_ids = np.where(clustering.labels_ == 1)
        logging.info(f"malicious: %s", malicious_ids)
        # malicious_ids = min(weights_norm[clustering.labels_ == 0],weights_norm[clustering.labels_ == 1] )
    # logging.info(f"cosine similarity: %s: ",F.cosine_similarity(cluster_mean_smaller, cluster_mean_larger, dim=0))
    # calculate clean weights
    clean_weights = []
    for i, weight in enumerate(weights_attacked):
        if i not in malicious_ids:
            clean_weights.append(weight)
    return malicious_ids, clean_weights


def fl_filter(baseline_weights, weights_attacked, deltas_attacked, received_ids):
    # self consistency for pred
    flattened_weights = flatten_weights(weights_attacked)
    # download dictionary from history
    file_path = "./flfilter_records.pkl"
    if os.path.exists(file_path):
        logging.info(f"flfilter is loading parameters from file.")
        with open(file_path, "rb") as file:
            last_weights_dict = pickle.load(file)
    else:
        # Initilization
        last_weights_dict = {
            client_id + 1: torch.zeros(len(flattened_weights[0]))
            for client_id in range(Config().clients.total_clients)
        }  # torch.zeros(len(flattened_weights[0]))

    # get last weights for received client from all_dict and make it as list
    last_weights = []
    for received_id in received_ids:
        last_weights.append(last_weights_dict[received_id])
    last_weights = torch.stack(last_weights)

    # make pre
    alpha = 0.1  # could be adaptive
    logging.info(f"last weights dictionary: %s", last_weights_dict)
    logging.info(f"last_weights: %s", last_weights)
    logging.info(f"flattened weights (targeted): %s", flattened_weights)
    self_consistency_pre = alpha * last_weights + (1 - alpha) * flattened_weights
    logging.info(f"self consistency pre: %s", self_consistency_pre)
    # group consistency
    # group average
    group_consistency_avg = torch.mean(flattened_weights, dim=0)  # axis?
    logging.info(f"group consistency avg: %s", group_consistency_avg)
    # group pre
    group_consistency_pre = 0.5 * group_consistency_avg + 0.5 * flattened_weights
    logging.info(f"group consistency pre: %s", group_consistency_pre)
    # joint pre
    beta = 0.5
    joint_pre = beta * self_consistency_pre + (1 - beta) * group_consistency_pre
    logging.info(f"joint_pre: %s", joint_pre)
    # distancing
    logging.info(f"joint_pre - flattened_weights : %s", joint_pre - flattened_weights)
    distances = torch.norm((joint_pre - flattened_weights), dim=1)  # ** 2
    logging.info(f"distance: %s", distances)
    # threshold: if large, block it and drag into blacklist; middle, delay aggregation; small, classify into clean clients
    clean_ids = []
    malicious_ids = []
    suspicious_ids = []
    threshold1 = 30
    threshold2 = 10
    for i, dis in enumerate(distances):
        if dis > threshold1:
            malicious_ids.append(i)
        elif dis < threshold2:
            clean_ids.append(i)
        else:
            suspicious_ids.append(i)

    # get clean weights
    clean_weights = []
    for i, weights in enumerate(weights_attacked):
        if i in clean_ids:
            clean_weights.append(weights)

    # update history for all clients
    # or just use dictionary with client id as key instead of list
    # update the dictionary as self consistency history
    for index, received_id in enumerate(received_ids):  # self_consistency_pre):
        last_weights_dict[received_id] = self_consistency_pre[index]

    # save the update dictionary to .pkl file
    file_path = "./flfilter_records.pkl"
    with open(file_path, "wb") as file:
        pickle.dump(last_weights_dict, file)

    return malicious_ids, clean_weights


registered_detectors = {
    "FLDetector": fl_detector,
    "Spectral_anomaly": spectral_anomaly_detection,
    "MAB-RFL": mab_rfl,
    "FLFilter": fl_filter,
}
