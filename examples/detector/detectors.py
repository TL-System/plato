import torch
import logging
from plato.config import Config
from scipy.stats import norm
import numpy as np
from mxnet import nd
from sklearn.cluster import KMeans
import pickle
import os


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
        return registered_defence

    raise ValueError(f"No such defence: {detector_type}")


def lbfgs(
    weights_attacked, global_weights_record, gradients_record, last_weights
):  # args, S_k_list, Y_k_list, v):
    # Approximate integrated Hessian value based on weights
    # weights record; gradients record; weight-last_weight
    # global model after iteration. client model updates? fldetector uses gradient as the local epoch is 1, which is not realistic.
    # I'll use model update as gradients instead to test the performance. this would be a more practical scenario.

    # could we read from local file within this function?
    # load info into the following variable? main function only update the record(history)

    global_weights = torch.cat(
        global_weights_record, dim=1
    )  # curr_S_k=nd.concat(*S_k_list, dim=1)
    gradients = torch.cat(gradients_record)  # curr_Y_k = nd.concat(*Y_k_list, dim=1)
    global_times_gradients = torch.dot(global_weights.T, gradients)
    global_times_global = torch.dot(global_weights.T, global_weights)
    R_k = np.triu(global_times_gradients.asnumpy())
    L_k = global_times_gradients - torch.tensor(R_k)
    sigma_k = torch.dot(global_weights_record[-1].T, gradients_record[-1]) / (
        torch.dot(gradients_record[-1].T, gradients_record[-1])
    )
    D_k_diag = torch.diag(global_times_gradients)
    upper_mat = torch.cat(*[sigma_k * global_times_global, L_k], dim=1)
    lower_mat = torch.cat(*[L_k.T, -torch.diag(D_k_diag)], dim=1)
    mat = torch.cat(*[upper_mat, lower_mat], dim=0)
    mat_inv = torch.inverse(mat)

    v = weights_attacked - last_weights
    approx_prod = sigma_k * v
    p_mat = torch.cat(
        [torch.dot(global_weights.T, sigma_k * v), torch.dot(gradients.T, v)], dim=0
    )
    approx_prod -= torch.dot(
        torch.dot(torch.cat([sigma_k * global_weights, gradients], dim=1), mat_inv),
        p_mat,
    )

    return approx_prod


def gap_statistics(score):
    nrefs = 10
    ks = range(1, 8)
    gaps = np.zeros(len(ks))
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks))
    min = np.min(score)
    max = np.max(score)
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
    # print(gapDiff)
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
    if np.mean(score[label_pred == 0]) < np.mean(score[label_pred == 1]):
        # cluster with smaller mean value is clean clients
        clean_ids = np.where(label_pred == 0)[0]
        malicious_ids = np.where(label_pred == 1)[0]
    else:
        clean_ids = np.where(label_pred == 1)[0]
        malicious_ids = np.where(label_pred == 0)[0]
    return malicious_ids, clean_ids


def fl_detector( baseline_weights, weights_attacked, deltas_attacked):
    """https://arxiv.org/pdf/2207.09209.pdf"""
    # prediction: need a history: use local files?
    file_path = "./records.pkl"
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            global_weights_record = pickle.load(file)
            gradients_record = pickle.load(file)
            last_weights = pickle.load(file)
            last_gradients = pickle.load(file)
            malicious_score = pickle.load(file)

    # Approximation
    hvp = lbfgs(
        deltas_attacked, global_weights_record, gradients_record, last_weights
    )  

    # Make prediction by Cauchy mean value theorem
    pred_grad = []
    for i in range(len(last_gradients)):
        pred_grad.append(last_gradients[i] + hvp)

    # Calculate distance for scoring
    distance = torch.norm(
        (torch.cat(pred_grad, dim=1) - torch.cat(deltas_attacked, dim=1)), axis=0
    ).asnumpy()
    # normalize distance
    distance = distance / np.sum(distance)
    # add new distance score into malicious score record
    malicious_score = np.row_stack((malicious_score, distance))
    # clustering
    if malicious_score.shape[0] >= 11:
        if gap_statistics(np.sum(malicious_score[-10:], axis=0)):
            logging.info(f"malicious clients detected!")
            malicious_ids, clean_ids = detection(np.sum(malicious_score[-10:], axis=0))
            break

    # update record
    global_weights_record.append(baseline_weights)
    gradients_record.append(deltas_attacked)
    weights_attacked.append(weights_attacked)
    last_gradients = deltas_attacked
    # save into local file
    file_path = "./records.pkl"
    with open(file_path, "wb") as file:
        pickle.dump(global_weights_record, file)
        pickle.dump(gradients_record, file)
        pickle.dump(weights_attacked, file)
        pickle.dump(last_gradients, file)
        pickle.dump(malicious_score, file)

    # remove poisoned weights
    clean_weights=[]
    for i, weight in enumerate(weights_attacked):
        if i in clean_ids:
            clean_weights.append(weight)

    return clean_weights


registered_detectors = {
    "FLDetector": fl_detector,
}
