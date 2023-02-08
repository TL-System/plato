"""
References:

Liu et al., "FedEraser: Enabling Efficient Client-Level Data Removal from Federated Learning Models,"
in IWQoS 2021.

Shokri et al., "Membership Inference Attacks Against Machine Learning Models," in IEEE S&P 2017.

https://ieeexplore.ieee.org/document/9521274
https://arxiv.org/pdf/1610.05820.pdf
"""
import logging

import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.nn.functional import softmax

from plato.config import Config


def train_attack_model(shadow_model, in_dataloader, out_dataloader):
    """Train attack model with shadow models."""
    logging.info("Training attack model")

    pred_4_mem = torch.zeros([1, Config().data.num_classes])
    pred_4_mem = pred_4_mem.to(Config().device())
    with torch.no_grad():
        for _, (data, _) in enumerate(in_dataloader):
            data = data.to(Config().device())
            out = shadow_model(data)
            pred_4_mem = torch.cat([pred_4_mem, out])
    pred_4_mem = pred_4_mem[1:, :]
    pred_4_mem = softmax(pred_4_mem, dim=1)
    pred_4_mem = pred_4_mem.cpu()
    pred_4_mem = pred_4_mem.detach().numpy()

    pred_4_nonmem = torch.zeros([1, Config().data.num_classes])
    pred_4_nonmem = pred_4_nonmem.to(Config().device())
    with torch.no_grad():
        for _, (data, _) in enumerate(out_dataloader):
            data = data.to(Config().device())
            out = shadow_model(data)
            pred_4_nonmem = torch.cat([pred_4_nonmem, out])
    pred_4_nonmem = pred_4_nonmem[1:, :]
    pred_4_nonmem = softmax(pred_4_nonmem, dim=1)
    pred_4_nonmem = pred_4_nonmem.cpu()
    pred_4_nonmem = pred_4_nonmem.detach().numpy()

    attacker = CatBoostClassifier(
        iterations=200,
        depth=2,
        learning_rate=0.5,
        loss_function="Logloss",
        verbose=False,
    )

    att_X = np.vstack((pred_4_mem, pred_4_nonmem))
    att_X.sort(axis=1)
    att_y = np.hstack((np.ones(pred_4_mem.shape[0]), np.zeros(pred_4_nonmem.shape[0])))
    att_y = att_y.astype(np.int16)

    X_train, X_test, y_train, y_test = train_test_split(att_X, att_y, test_size=0.1)

    attacker.fit(X_train, y_train)

    return attacker


def launch_attack(target_model, attack_model, attack_dataloader, out_dataloader):
    """Launch attack toward the target model."""
    logging.info("Launching attack")

    # The posteriors of unlearned/forgotten data through the target model
    unlearn_X = torch.zeros([1, Config().data.num_classes])
    unlearn_X = unlearn_X.to(Config().device())
    with torch.no_grad():
        for _, (data, _) in enumerate(attack_dataloader):
            data = data.to(Config().device())
            out = target_model(data)
            unlearn_X = torch.cat([unlearn_X, out])

    unlearn_X = unlearn_X[1:, :]
    unlearn_X = softmax(unlearn_X, dim=1)
    unlearn_X = unlearn_X.cpu().detach().numpy()

    unlearn_X.sort(axis=1)
    unlearn_y = np.ones(unlearn_X.shape[0])
    unlearn_y = unlearn_y.astype(np.int16)

    N_unlearn_sample = len(unlearn_y)

    # The posteriors of testset data through the target model
    test_X = torch.zeros([1, Config().data.num_classes])
    test_X = test_X.to(Config().device())
    with torch.no_grad():
        for _, (data, _) in enumerate(out_dataloader):
            data = data.to(Config().device())
            out = target_model(data)
            test_X = torch.cat([test_X, out])

            if test_X.shape[0] > N_unlearn_sample:
                break
    test_X = test_X[1 : N_unlearn_sample + 1, :]
    test_X = softmax(test_X, dim=1)
    test_X = test_X.cpu().detach().numpy()

    test_X.sort(axis=1)
    test_y = np.zeros(test_X.shape[0])
    test_y = test_y.astype(np.int16)

    # The balanced data set consists of 50% training 50% testing
    XX = np.vstack((unlearn_X, test_X))
    YY = np.hstack((unlearn_y, test_y))

    pred_YY = attack_model.predict(XX)
    acc = accuracy_score(YY, pred_YY)
    pre = precision_score(YY, pred_YY, pos_label=1)
    rec = recall_score(YY, pred_YY, pos_label=1)
    print("MIA Attacker accuracy = {:.4f}".format(acc))
    print("MIA Attacker precision = {:.4f}".format(pre))
    print("MIA Attacker recall = {:.4f}".format(rec))
