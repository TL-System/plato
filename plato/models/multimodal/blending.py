#!/usr/bin/env python
# -*- coding: utf-8 -*-

# the overfitting value is the gap between the train loss L_i^T and
#   the groundtruth L_i^* w.r.t the hypothetical target distribution
#   Note: the  L^* is approximated by the validation loss L^V


# We define overfitting at epoch N as the gap between LTN and L∗N (approximated by ON in fig. 3).
def compute_overfitting_O(eval_avg_loss, train_avg_loss):
    return eval_avg_loss - train_avg_loss


# n < N
def compute_delta_overfitting_O(n_eval_avg_loss, n_train_avg_loss,
                                N_eval_avg_loss, N_train_avg_loss):
    delta_O = compute_overfitting_O(n_eval_avg_loss,
                                    n_train_avg_loss) - compute_overfitting_O(
                                        N_eval_avg_loss, N_train_avg_loss)
    return delta_O


def compute_generalization_G(eval_avg_loss):
    return eval_avg_loss


def compute_delta_generalization(eval_avg_loss_n, eval_avg_loss_N):
    return compute_generalization_G(
        eval_avg_loss_n) - compute_generalization_G(eval_avg_loss_N)


# n < N,
def OGR_n2N(n_eval_avg_loss, n_train_avg_loss, N_eval_avg_loss,
            N_train_avg_loss):
    """ Compute the OGR = abs(delta_O/delta_G)"""
    delta_O = compute_delta_overfitting_O(n_eval_avg_loss, n_train_avg_loss,
                                          N_eval_avg_loss, N_train_avg_loss)
    delta_G = compute_delta_generalization(n_eval_avg_loss, N_eval_avg_loss)

    ogr = abs(delta_O / delta_G)
    return ogr
