"""

the overfitting value is the gap between the train loss L_i^T and
the groundtruth L_i^* w.r.t the hypothetical target distribution
Note: the  L^* is approximated by the validation loss L^V

"""
import numpy as np


def compute_overfitting_o(eval_avg_loss, train_avg_loss):
    """ We define overfitting at epoch N as the gap between LTN and
        L∗N (approximated by ON in fig. 3). """
    return eval_avg_loss - train_avg_loss


def compute_delta_overfitting_o(n_eval_avg_loss, n_train_avg_loss,
                                N_eval_avg_loss, N_train_avg_loss):
    """ Compute the overfitting O based on losses between step n and N, (n < N) """
    delta_O = compute_overfitting_o(n_eval_avg_loss,
                                    n_train_avg_loss) - compute_overfitting_o(
                                        N_eval_avg_loss, N_train_avg_loss)
    return delta_O


def compute_generalization_g(eval_avg_loss):
    """ Compute the generalization g which is actually the evluation loss """
    return eval_avg_loss


def compute_delta_generalization(eval_avg_loss_n, eval_avg_loss_N):
    """ Compute the difference of the generalization """
    return compute_generalization_g(
        eval_avg_loss_n) - compute_generalization_g(eval_avg_loss_N)


# n < N,
def OGR_n2N(n_eval_avg_loss, n_train_avg_loss, N_eval_avg_loss,
            N_train_avg_loss):
    """ Compute the OGR = abs(delta_O/delta_G)"""
    delta_O = compute_delta_overfitting_o(n_eval_avg_loss, n_train_avg_loss,
                                          N_eval_avg_loss, N_train_avg_loss)
    delta_G = compute_delta_generalization(n_eval_avg_loss, N_eval_avg_loss)

    ogr = abs(delta_O / delta_G)
    return ogr


# Optimal Gradient Blend
#   x << N
def get_optimal_gradient_blend_weights(modalities_losses_n,
                                       modalities_losses_N):
    """ Get the weights of modaliteis for optimal gradient blending

        Args:
            modalities_losses_n (dict): contains the train/eval losses for each modality in epoch n
            modalities_losses_N (dict): contains the train/eval losses for each modality in epoch N

            The structure of the above two dicts should be: (for example)
                {"train": {"RGB": float, "Flow": float},
                "eval": {"RGB": float, "Flow": float}}

        The equation:
            w^i = <∇L^*, v_i>/(σ_i)^2 * 1/Z
                = <∇L^*, v_i>/(σ_i)^2 * 1/(sum_i <∇L^*, v_i>/2*(σ_i)^2)
                = G^i / (O^i)^2 * 1 / (sum_i G^i / (2 * (O^i)^2))

            where G^i = G_N,n =  L^*_n − L^*_N = compute_delta_generalization,
                    O^i = O_N,n =  O_N - O_n = compute_delta_overfitting_O
    """
    modality_names = list(modalities_losses_n["train"].keys())

    Z = 0
    modls_GO = dict()
    for modality_nm in modality_names:
        modl_eval_avg_loss_n = modalities_losses_n["eval"][modality_nm]
        modl_subtrain_avg_loss_n = modalities_losses_n["train"][modality_nm]
        modl_eval_avg_loss_N = modalities_losses_N["eval"][modality_nm]
        modl_subtrain_avg_loss_N = modalities_losses_N["train"][modality_nm]
        G_i = compute_delta_generalization(
            eval_avg_loss_n=modl_eval_avg_loss_n,
            eval_avg_loss_N=modl_eval_avg_loss_N)
        O_i = compute_delta_overfitting_o(
            n_eval_avg_loss=modl_eval_avg_loss_n,
            n_train_avg_loss=modl_subtrain_avg_loss_n,
            N_eval_avg_loss=modl_eval_avg_loss_N,
            N_train_avg_loss=modl_subtrain_avg_loss_N)

        modls_GO[modality_nm] = G_i / (O_i * O_i)
        Gi_div_sqr_Oi = G_i / (2 * O_i * O_i)

        Z += Gi_div_sqr_Oi

    optimal_weights = dict()
    for modality_nm in modality_names:
        optimal_weights[modality_nm] = modls_GO[modality_nm] / Z

    return optimal_weights


# Optimal Gradient Blend
#   x << N
def get_optimal_gradient_blend_weights_og(delta_OGs):
    """ Get the weights of clients for optimal gradient blending

        Args:
            delta_OGs (list): each item is a tuple that contains (delta_O, delta_G)

            The structure of the above two dicts should be: (for example)
                [(0.2, 0.45), (0.3, 0.67)]

        The equation that is the same as the weights computation for modalities:
            w^i = <∇L^*, v_i>/(σ_i)^2 * 1/Z
                = <∇L^*, v_i>/(σ_i)^2 * 1/(sum_i <∇L^*, v_i>/2*(σ_i)^2)
                = G^i / (O^i)^2 * 1 / (sum_i G^i / (2 * (O^i)^2))

            where G^i = G_N,n =  delta_G,
                    O^i = O_N,n =  O_N - O_n = delta_O
    """
    num_of_clients = len(delta_OGs)

    Z = 0
    clients_ratios = list()
    for cli_i in range(num_of_clients):
        cli_delta_O, cli_delta_G = delta_OGs[cli_i]

        G_i = cli_delta_G
        O_i = cli_delta_O

        #models_GO[modality_nm] = G_i / (O_i * O_i)
        Gi_div_sqr_Oi = G_i / (2 * O_i * O_i)

        clients_ratios.append(Gi_div_sqr_Oi)

        Z += Gi_div_sqr_Oi

    optimal_weights = np.array(clients_ratios) / Z

    return optimal_weights
