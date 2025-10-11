import logging
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering, KMeans

from plato.config import Config

# Configure logging
logging.basicConfig(filename="app.log", filemode="w", level=logging.INFO)


def get():
    detector_type = (
        Config().server.detector_type
        if hasattr(Config().server, "detector_type")
        else None
    )

    if detector_type is None:
        logging.info("No defence is applied.")
        return None

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
    # Approximate integrated Hessian value
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

    upper_mat = torch.cat(((sigma_k * global_times_global), L_k), dim=1)
    lower_mat = torch.cat((L_k.T, -torch.diag(D_k_diag)), dim=1)
    mat = torch.cat((upper_mat, lower_mat), dim=0)
    mat_inv = torch.inverse(mat)

    v = weights_attacked - last_weights  # deltas_attacked from selected clients
    v = torch.mean(v, dim=0)
    approx_prod = sigma_k * v
    p_mat = torch.cat(
        (torch.matmul(global_weights, (sigma_k * v).T), torch.matmul(gradients, v.T)),
        dim=0,
    )
    approx_prod -= torch.matmul(
        torch.matmul(
            torch.cat((sigma_k * global_weights.T, gradients.T), dim=1), mat_inv
        ),
        p_mat,
    ).T
    return approx_prod


def gap_statistics(score):
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
    estimator = KMeans(n_clusters=2)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    # Print the members in each cluster
    for cluster in np.unique(label_pred):
        cluster_members = score[label_pred == cluster]
        logging.info(f"Cluster {cluster + 1} members: %s", cluster_members)
    if np.mean(score[label_pred == 0]) < np.mean(score[label_pred == 1]):
        # cluster with smaller mean value is clean clients
        clean_ids = np.where(label_pred == 0)[0]
        malicious_ids = np.where(label_pred == 1)[0]
    else:
        clean_ids = np.where(label_pred == 1)[0]
        malicious_ids = np.where(label_pred == 0)[0]
    # logging.info(f"clean_ids: %s", clean_ids)
    return malicious_ids, clean_ids


def detection_cos(score):
    estimator = KMeans(n_clusters=3)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    # Print the members in each cluster
    for cluster in np.unique(label_pred):
        cluster_members = score[label_pred == cluster]
        logging.info(f"Cluster {cluster + 1} members: %s", cluster_members)
    if (np.mean(score[label_pred == 0]) > np.mean(score[label_pred == 1])) and (
        np.mean(score[label_pred == 0]) > np.mean(score[label_pred == 2])
    ):
        # cluster with larger value is attacker
        clean_ids = np.concatenate(
            (np.where(label_pred == 1)[0], np.where(label_pred == 2)[0])
        )
        malicious_ids = np.where(label_pred == 0)[0]
    elif (np.mean(score[label_pred == 1]) > np.mean(score[label_pred == 0])) and (
        np.mean(score[label_pred == 1]) > np.mean(score[label_pred == 2])
    ):
        clean_ids = np.concatenate(
            (np.where(label_pred == 0)[0], np.where(label_pred == 2)[0])
        )
        malicious_ids = np.where(label_pred == 1)[0]
    else:
        clean_ids = np.concatenate(
            (np.where(label_pred == 1)[0], np.where(label_pred == 0)[0])
        )
        malicious_ids = np.where(label_pred == 2)[0]

    return malicious_ids, clean_ids


def pre_data_for_visualization(deltas_attacked, received_staleness):
    # saved received local deltas for round x
    # logging.info(f"starting preparing data for visualization")
    flattened_deltas_attacked = flatten_weights(deltas_attacked)
    # list to torch tensor
    received_staleness = torch.tensor(received_staleness)

    model_path = Config().params["model_path"]
    model_name = Config().trainer.model_name

    try:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
    except FileExistsError:
        pass

    try:
        # List all files and directories in the given folder
        items = os.listdir(model_path)

        # Count the number of files (ignoring directories)
        file_count = sum(
            1 for item in items if os.path.isfile(os.path.join(model_path, item))
        )

        file_count = str(
            file_count + 1
        )  # plus one so can be directly used in the following code when create folder for each communication round

    except Exception as e:
        pass

    # logging.info(f"saving reveived deltas...")
    file_path = f"{model_path}/" + file_count + ".pkl"
    with open(file_path, "wb") as file:
        pickle.dump(flattened_deltas_attacked, file)
        pickle.dump(received_staleness, file)

    logging.info(
        "[Server #%d] Model saved to %s at round %s.",
        os.getpid(),
        file_path,
        file_count,
    )


def async_filter(
    baseline_weights,
    weights_attacked,
    deltas_attacked,
    received_ids,
    received_staleness,
):
    # first group clients based on their staleness
    staleness_bound = Config().server.staleness_bound
    flattened_weights_attacked = flatten_weights(weights_attacked)

    # only for visualization
    # pre_data_for_visualization(deltas_attacked, received_staleness)

    file_path = "./async_record" + str(os.getpid()) + ".pkl"
    if os.path.exists(file_path):
        logging.info(f"loading parameters from file.")
        with open(file_path, "rb") as file:
            global_weights_record = pickle.load(file)
            global_num_record = pickle.load(file)
    else:
        global_weights_record = []
        global_num_record = []

    weight_groups = {i: [] for i in range(20)}
    id_groups = {i: [] for i in range(20)}
    for i, (staleness, weights) in enumerate(
        zip(received_staleness, flattened_weights_attacked)
    ):
        weight_groups[staleness].append(weights)
        id_groups[staleness].append(i)

    # calcuate cos_similarity within a group and identify statistical outliers
    all_mali_ids = []
    avg_current = torch.zeros_like(torch.mean(weights, dim=0))
    num_current = 0
    for staleness, weights in weight_groups.items():
        if len(weights) >= 3:
            weights = torch.stack(weights)
            # find out avg at the same round
            if staleness == 0:
                avg = torch.mean(weights, dim=0) + 1e-10
                avg_current = avg
                num_current = len(weight_groups[0])
            else:
                avg = (
                    global_weights_record[-1 * staleness]
                    * global_num_record[-1 * staleness]
                    + torch.mean(weights, dim=0) * len(weight_groups[staleness])
                ) / (global_num_record[-1 * staleness] + len(weight_groups[staleness]))
                # update record
                global_weights_record[-1 * staleness] = avg
                global_num_record[-1 * staleness] += len(weight_groups[staleness])

            similarity = F.cosine_similarity(weights, avg).numpy()
            logging.info(f"Group %d cosine similarity: %s", staleness, similarity)
            # whether or not to normalization is a question

            distance = torch.norm((avg - weights), dim=1).numpy()

            distance = distance / np.sum(distance)  # normalization

            logging.info("applying 3 clustering for comparison")
            malicious_ids2, clean_ids = detection_cos(distance)

            malicious_ids = malicious_ids2

            malicious_ids = [id_groups[staleness][id] for id in malicious_ids.tolist()]
            logging.info(f"malicious in this group: %s", malicious_ids)
            all_mali_ids.extend(malicious_ids)

    # save into local file
    global_weights_record.append(avg_current)
    global_num_record.append(num_current)
    file_path = "./async_record" + str(os.getpid()) + ".pkl"
    with open(file_path, "wb") as file:
        pickle.dump(global_weights_record, file)
        pickle.dump(global_num_record, file)

    # remove suspecious weights
    clean_weights = []
    for i, weight in enumerate(weights_attacked):
        if i not in all_mali_ids:
            clean_weights.append(weight)

    return all_mali_ids, clean_weights


def fl_detector(
    baseline_weights,
    weights_attacked,
    deltas_attacked,
    received_ids,
    received_staleness,
):
    # https://arxiv.org/pdf/2207.09209.pdf
    # torch.set_printoptions(threshold=10**8)
    # malicious_ids_list = [] # for test case only, will be remove after finished

    clean_weights = weights_attacked

    flattened_weights_attacked = flatten_weights(weights_attacked)
    flattened_baseline_weights = flatten_weight(baseline_weights)
    flattened_deltas_attacked = flatten_weights(deltas_attacked)
    id_temp = [x - 1 for x in received_ids]
    local_update_current = torch.stack(
        [x[1] for x in sorted(zip(id_temp, flattened_deltas_attacked))]
    )

    window_size = 0
    malicious_ids = []

    file_path = "./record" + str(os.getpid()) + ".pkl"
    if os.path.exists(file_path):
        logging.info(f"loading parameters from file.")
        with open(file_path, "rb") as file:
            # download dict from file
            # below are records for fldetector
            global_weights_record = pickle.load(file)
            gradients_record = pickle.load(file)
            last_weights = pickle.load(file)
            last_gradients = pickle.load(file)
            malicious_score_dict = pickle.load(file)

        # get weights for received clients at this round only
        malicious_score = []
        # last_move = []
        for received_id in received_ids:
            # last_move.append(last_move_dict[received_id]) # for avg
            malicious_score.append(
                list(filter(lambda x: x is not None, malicious_score_dict[received_id]))
            )  # for fldetector

        moving_avg = torch.mean(flattened_deltas_attacked, dim=0) + 1e-10  # for avg

        if len(global_weights_record) >= window_size + 1:
            """Below are fldetector"""
            # Make predication by Cauchy mean value theorem in fldetector
            hvp = lbfgs(
                flattened_weights_attacked,
                global_weights_record,
                gradients_record,
                last_weights,
            )

            pred_grad = torch.add(last_gradients, hvp)

            # Calculate distance for scoring
            distance1 = torch.norm(
                (pred_grad - flattened_deltas_attacked), dim=1
            ).numpy()
            logging.info(f"distance in fldetectors before normalization: %s", distance1)
            # Normalize distance
            distance1 = distance1 / np.sum(distance1)
            logging.info(f"the fldetector distance after normalization: %s", distance1)

            # add new distance score into malicious score record
            # moving averaging
            malicious_score_current = []
            for scores, dist in zip(malicious_score, distance1):
                scores.append(dist)
                score = sum(scores) / len(scores)
                malicious_score_current.append(score)
            logging.info(
                f"the malicious score current round: %s", malicious_score_current
            )
            # cluserting and detection (smaller score represents benign clients)
            malicious_ids, clean_ids = detection(
                np.array(malicious_score_current)
            )  # np.sum(malicious_score[-10:], axis=0))
            logging.info(f"fldetector malicious ids: %s", malicious_ids)

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
        malicious_score_dict = {
            client_id + 1: [] for client_id in range(Config().clients.total_clients)
        }  # len(received_ids)*[0]
        distance1 = len(received_ids) * [None]  # none to keep update consistent

    # update record
    for index, received_id in enumerate(received_ids):  # self_consistency_pre):
        malicious_score_dict[received_id].append(
            distance1[index]
        )  # distance should be initialized (this one is for fldetector)

    global_weights_record.append(flattened_baseline_weights - last_weights)
    gradients_record.append(
        torch.mean(flattened_deltas_attacked, dim=0) - last_gradients
    )
    last_weights = flattened_baseline_weights
    last_gradients = torch.mean(flattened_deltas_attacked, dim=0)

    # save into local file
    file_path = "./record" + str(os.getpid()) + ".pkl"
    with open(file_path, "wb") as file:
        pickle.dump(global_weights_record, file)
        pickle.dump(gradients_record, file)
        pickle.dump(last_weights, file)
        pickle.dump(last_gradients, file)
        pickle.dump(malicious_score_dict, file)
    logging.info(f"malicious_ids: %s", malicious_ids)
    return malicious_ids, clean_weights


registered_detectors = {
    "FLDetector": fl_detector,
    "AsyncFilter": async_filter,
}
