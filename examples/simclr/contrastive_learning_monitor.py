"""
The implementation of monitor for contrastive learning.

As the training phase of contrastive learning does not output general metrics
 , such as accuracy, but only the losses, it is highly required to monitor the
 training process by other external methods.

Then motivated by the insight that the encoder of contrastive learning methods 
 produces high-quality, such as distinguishable, representation, we can monior
 the generated representations based on the cluster methods, such as KNN.

"""

from tqdm import tqdm

import torch.nn.functional as F
import torch


def knn_monitor(encoder,
                memory_data_loader,
                test_data_loader,
                device,
                k=200,
                t=0.1,
                hide_progress=False):
    """ Using the KNN monitor to test the representation quality.
    
        This part of code is obtained from the official code for SimClr:
        https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N.

        Args:
            encoder (torch.nn.module): the defined encoder 
            memory_data_loader: the data loader of the trainset using the test 
                data augmentation
            test_data_loader:  the data loader of the testset using the test 
                data augmentation

     """
    encoder.eval()
    classes = len(memory_data_loader.dataset.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader,
                                 desc='Feature extracting',
                                 leave=False,
                                 disable=hide_progress):
            data, target = data.to(device), target.to(device)
            feature = encoder(data)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(
            memory_data_loader.dataset.dataset.targets,
            device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
        for data, target in test_bar:
            data, target = data.to(device), target.to(device)
            feature = encoder(data)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels,
                                      classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix({'Accuracy': total_top1 / total_num})
    return total_top1 / total_num


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1),
                              dim=-1,
                              index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k,
                                classes,
                                device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1,
                                          index=sim_labels.view(-1, 1),
                                          value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) *
                            sim_weight.unsqueeze(dim=-1),
                            dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
