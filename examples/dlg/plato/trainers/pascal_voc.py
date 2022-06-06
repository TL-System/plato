"""
A customized trainer for image segmentation on PASCAL VOC dataset (2012).
"""
import torch.nn as nn
import torch
import numpy as np

from plato.trainers import basic


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class, ) * 2)

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(
            self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) +
                                      np.sum(self.confusion_matrix, axis=0) -
                                      np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, ) * 2)


class Trainer(basic.Trainer):
    """The federated learning trainer for the image segmentation on PASCAL VOC"""
    def __init__(self, model=None):
        """Initializing the trainer with the provided model.

        Arguments:
        model: The model to train.
        client_id: The ID of the client using this trainer (optional).
        """
        super().__init__(model)

        self.loss_criterion = nn.BCEWithLogitsLoss()
        self.num_class = 20

    def test_model(self, config, testset):
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=config['batch_size'], shuffle=False)

        total = 0
        evaluator = Evaluator(self.num_class)
        evaluator.reset()
        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(
                    self.device)

                outputs = self.model(examples)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                labels = torch.squeeze(labels, 1).cpu().numpy()
                predicted = predicted.cpu().numpy()
                print('shape of pred: ', predicted.shape)
                print('shape of labels: ', labels.shape)
                evaluator.add_batch(labels, predicted)

            accuracy = evaluator.Mean_Intersection_over_Union()

        return accuracy
