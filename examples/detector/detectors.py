import torch
import logging
from plato.config import Config
from scipy.stats import norm
import numpy as np
from sklearn.cluster import KMeans
import pickle
import os
import gc


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

def lbfgs(
    deltas_attacked, global_weights_record, gradients_record, last_weights
):  # args, S_k_list, Y_k_list, v):
    # Approximate integrated Hessian value based on weights
    # weights record; gradients record; weight-last_weight
    # global model after iteration. client model updates? fldetector uses gradient as the local epoch is 1, which is not realistic.
    # I'll use model update as gradients instead to test the performance. this would be a more practical scenario.

    # could we read from local file within this function?
    # load info into the following variable? main function only update the record(history)
    # Check if GPU is available
   

    global_weights = torch.stack(global_weights_record)
    gradients = torch.stack(gradients_record)
    global_times_gradients = torch.matmul(global_weights.T, gradients)
    global_times_global = torch.matmul(global_weights.T, global_weights)
    R_k = np.triu(global_times_gradients.numpy())
    L_k = (global_times_gradients - torch.tensor(R_k))
    sigma_k = torch.matmul(torch.transpose(global_weights_record[-1],0,-1), gradients_record[-1]) / (
        torch.matmul(torch.transpose(gradients_record[-1],0,-1), gradients_record[-1])
    )
    logging.info(f"sigma_k: %s",sigma_k)
    D_k_diag = torch.diag(global_times_gradients)
    
    del R_k
    gc.collect()
    del global_times_gradients
    gc.collect()
    del global_weights_record
    gc.collect()
    del gradients_record
    gc.collect()


    logging.info(f"DK diag %s", D_k_diag)
    #logging.info(f"sigma_ka*global times global: %s",sigma_k*global_times_global)
    logging.info(f"L_k: %s",L_k)
    #logging.info(f"L_k.T shape: %s",L_k.shape)
    logging.info(f"sigma_k*global_times_global.T shape: %s",(global_times_global).shape)

    upper_mat = (sigma_k * global_times_global)
    del global_times_global
    gc.collect()
    logging.info(f"memory: %s",torch.cuda.memory_summary())
    upper_mat = torch.cat((upper_mat, L_k), dim=1)
   
    logging.info(f"upper_mat: %s", upper_mat)
    logging.info(f"L_k.T shape: %s",L_k.T.shape)


    logging.info(f"D_k_diag shape: %s",torch.diag(D_k_diag).shape)

    lower_mat = torch.cat((L_k.T, -torch.diag(D_k_diag)), dim=1)
    del L_k
    gc.collect()
    del D_k_diag
    gc.collect()

    logging.info(f"lower_mat: %s", lower_mat)
    mat = torch.cat((upper_mat, lower_mat), dim=0)
    logging.info(f"mat: %s", mat)
    mat_inv = torch.inverse(mat)
    logging.info(f"mat_inv: %s",mat_inv)

    v = deltas_attacked - last_weights
    logging.info(f"v: %s",v)
    approx_prod = sigma_k * v
    p_mat = torch.cat(
        (torch.matmul(global_weights.T, sigma_k * v), torch.matmul(gradients.T, v)), dim=0
    )
    logging.info(f"p_mat: ",p_mat)
    approx_prod -= torch.matmul(
        torch.matmul(torch.cat((sigma_k * global_weights, gradients), dim=1), mat_inv),
        p_mat,
    )


    return approx_prod


def gap_statistics(score):
    logging.info(f"in gap function")
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
    logging.info(f"line 105")
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


def fl_detector( baseline_weights, weights_attacked, deltas_attacked):
    """https://arxiv.org/pdf/2207.09209.pdf"""
    # flatten inputs
    flattened_weights = flatten_weights(weights_attacked)
    baseline_weights = flatten_weight(baseline_weights)
    deltas_attacked = flatten_weights(deltas_attacked)

    malicious_ids = []
    clean_weights = weights_attacked

    # prediction: need a history: use local files?
    file_path = "./records.pkl"
    if os.path.exists(file_path):
        logging.info(f"fldetector loading parameters from file.")
        with open(file_path, "rb") as file:
            global_weights_record = pickle.load(file)
            gradients_record = pickle.load(file)
            last_weights = pickle.load(file)
            last_gradients = pickle.load(file)
            malicious_score = pickle.load(file)

        if len(global_weights_record)>=3: # window

            logging.info(f"line 151 before calculating hvp")
            # Approximation
            hvp = lbfgs(
                deltas_attacked, global_weights_record, gradients_record, last_gradients
            )  
            logging.info(f"line 156: the hvp is %s", hvp)
            # Make prediction by Cauchy mean value theorem
            pred_grad = []
            
            # this should be local gradients
            for i in range(len(last_gradients)):
                pred_grad.append(last_gradients[i] + hvp)
            logging.info(f"the pred_grad is: %s", pred_grad)

            # Calculate distance for scoring
            distance = torch.norm(
                (torch.cat(pred_grad, dim=0) - torch.cat(deltas_attacked, dim=0)), axis=0
            ).numpy()
            logging.info(f"the distance is: %s", distance)
            # normalize distance
            distance = distance / np.sum(distance)
            logging.info(f"the distance after normalization: %s", distance)
            # add new distance score into malicious score record
            malicious_score = np.row_stack((malicious_score, distance))
            logging.info(f"the malicious score: %s", malicious_score)
            # clustering
            if malicious_score.shape[0] >= 11:
                logging.info(f"line 176 for checkpoint")
                if gap_statistics(np.sum(malicious_score[-10:], axis=0)):
                    logging.info(f"malicious clients detected!")
                    malicious_ids, clean_ids = detection(np.sum(malicious_score[-10:], axis=0))
        
            # remove poisoned weights
            clean_weights=[]
            for i, weight in enumerate(flattened_weights):
                if i in clean_ids:
                    clean_weights.append(weight)
        
    else: 
        logging.info(f"initializing fl parameter record")
        global_weights_record = []
        gradients_record = []
        last_gradients = torch.zeros(len(baseline_weights))
        last_weights = torch.zeros(len(baseline_weights))
        malicious_score = []
    
    # update record
    logging.info(f"line 183: updating record")
    global_weights_record.append(baseline_weights-last_weights)
    logging.info(f"len %d",len(global_weights_record))
    gradients_record.append(torch.mean(deltas_attacked, dim=0)-last_gradients)
    last_weights = baseline_weights 
    last_gradients = torch.mean(deltas_attacked,dim=0)
        
    # save into local file
    file_path = "./records.pkl"
    with open(file_path, "wb") as file:
        pickle.dump(global_weights_record, file)
        pickle.dump(gradients_record, file)
        pickle.dump(last_weights, file)
        pickle.dump(last_gradients, file)
        pickle.dump(malicious_score, file)

    return malicious_ids, clean_weights


registered_detectors = {
    "FLDetector": fl_detector,
}
