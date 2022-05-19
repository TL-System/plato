""" 
The implementation of wrapper for the contrastive dataset

"""

import numpy as np
import torch


class ContrastiveDataWrapper(torch.utils.data.Dataset):
    """Prepares the contrastive dataset for use in the self-supervised learning.
    
        This is the most common method used to prepare the contrastive sample.

        The positive sample is obtained by sampling one sample, except the given sample,
            from the label class of the given sample.
        
        The negative sample is obtained by sampling from a label class that is different
            from the label class of the given sample.
    
    """

    def __init__(self, dataset):
        self.dataset = dataset

        # all labels for samples of the dataset
        self.dataset_labels = self.dataset.targets
        # the category of labels
        self.unique_label = list(set(self.dataset_labels))
        self.label_classes_size = len(self.unique_label)

        # the samples index containing in each label
        self.label_indexs_pool = {
            label_id: np.where(np.array(self.dataset_labels) == label_id)[0]
            for label_id in self.unique_label
        }
        self.label_indexs_pool_size = {
            label_id: len(self.label_indexs_pool[label_id])
            for label_id in self.label_indexs_pool
        }

        # the number of samples selected in each label
        self.label_selected_index_count = {
            label_id: 0
            for label_id in self.unique_label
        }
        # the pointer sliding among unique_label, thus to
        #  present the samples within which class will be sampled
        #  to behave as the negative sample
        self.selected_label_count = 0

        # whether to prepare a positive sample
        self.is_positive = 0

    def __len__(self):
        return len(self.dataset)

    def increase_selected_label_count(self):
        """ Increase the selected_label_count by one, but with the upper bound. """
        self.selected_label_count += 1

        if self.selected_label_count == len(self.label_classes_size):
            self.selected_label_count = 0

    def increase_selected_index_count(self, target_label):
        """ increase the label_selected_index_count by one, but with the upper bound """
        # increase the selected samples count for this label
        self.label_selected_index_count[target_label] += 1
        upper_bound = self.label_indexs_pool_size[target_label]
        if self.label_selected_index_count[target_label] == upper_bound:
            self.label_selected_index_count[target_label] = 0

    def sample_label_pool(self, except_label):
        """ Sample one class of label from the label's class pool """
        # copy the label classes while removing the except_label
        temp_unique_label = self.unique_label
        temp_unique_label.remove(except_label)

        select_label_class_pos = self.selected_label_count % len(
            temp_unique_label)
        selected_label = temp_unique_label[select_label_class_pos]

        # increase the selected labels count
        self.increase_selected_label_count()

        return selected_label

    def sample_label_index_pool(
        self,
        target_label,
        except_sample_index=None,
    ):
        """ Sample one sample from the given label class. """

        # obtain all sample indexs for the target label
        label_index_pool = self.label_indexs_pool[target_label]
        if except_sample_index is not None:
            # remove the except sample index
            label_index_pool.remove(except_sample_index)

        # obtain how many samples have been selected
        label_index_selected_count = self.label_selected_index_count[
            target_label]
        # get the position of the one that has not been used
        positive_sample_pos = label_index_selected_count % len(
            label_index_pool)
        # obtain the sample's index
        sample_index = label_index_pool[positive_sample_pos]

        # increase the selected samples count for this label
        self.increase_selected_index_count()

        return sample_index

    def __getitem__(self, item_index):
        """ generate the contrastive dataset """

        # the label for the obtained paired
        #   contrastive samples
        # - 0, negative sample
        # - 1, positive sample
        paired_sample_label = 0

        # obtain one sample based on the required item index
        obtained_sample, sample_label = self.dataset[item_index]

        # decide to prepare positive or negative sample
        self.is_positive = np.random.randint(1, size=10) > 0.5

        if self.is_positive:
            # obtain positive sample by sampling from the same
            #   label class.
            positive_sample_index = self.sample_label_index_pool(
                sample_label, except_sample_index=item_index)

            paired_sample_label = 1
            prepared_sample, _ = self.dataset[positive_sample_index]

        else:
            # obtain negative sample by sampling from different
            #   label classes.
            selected_label = self.sample_label_pool(sample_label)
            negative_sample_index = self.sample_label_index_pool(
                selected_label)

            paired_sample_label = 0
            prepared_sample, _ = self.dataset[negative_sample_index]

        return tuple(obtained_sample, prepared_sample), paired_sample_label


class ContrastiveAugmentDataWrapper(torch.utils.data.Dataset):
    """Prepares the contrastive dataset for use in the self-supervised learning.

        The contrastive sample is prepared through the data augmentation method,
         such as the one used in work SimSiam [1].

        [1].  Chen & He, Exploring Simple Siamese Representation Learning, 2021.

        For the input sample x, it only prepares paired positive samples by 
            applying data augmentation method, i.e, aug, to x.
            x1, x2 = aug(x), aug(x) # random augmentation

        The difference between their implementation and ours is that they perform
         the preparation of a batch of samples obtained in the data loader
         while ours perform the preparation on a sample in the dataset.

         The pipeline is:   dataset -> dataset loader -> a batch of samples
    """

    def __init__(self, dataset, aug_transformer):
        self.dataset = dataset
        self.raw_data = self.dataset.data

        # all labels for samples of the dataset
        self.dataset_labels = self.dataset.targets

        # predefined transformer used as the augmentation
        self.aug_transformer = aug_transformer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item_index):
        """ generate the contrastive dataset.
        
        Note, in general, there is no need to output the sample label as 
         self-supervised learning relies on unannotated samples. 
         But we still inject the sample label into the output to make 
         it maintain consistency with other datasets object.
         """
        raw_sample = self.raw_data[item_index]
        sample_label = self.dataset_labels[item_index]

        # the aug_transformer is expected to output paired samples
        prepared_sample1, prepared_sample2 = self.aug_transformer(raw_sample)

        return tuple(prepared_sample1, prepared_sample2), sample_label
