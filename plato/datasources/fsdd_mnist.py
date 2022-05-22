"""
The FSDD-based MNIST dataset.

Please access the git repo "https://github.com/Jakobovski/free-spoken-digit-dataset"
 for the details of Free Spoken Digit Dataset (FSDD) dataset.

In summary, 6 speakers generate 3,000 recordings (50 of each digit per speaker).

Another one audio data source is the AudioMNIST:
 https://github.com/jayrodge/AudioMNIST-using-PyTorch.
 https://github.com/soerenab/AudioMNIST.

However, limited by time, we do not implement it (AudioMNIST-based MNIST) currently.

"""

import numpy as np
from plato.config import Config
from plato.datasources import multimodal_base
from torchaudio.transforms import MFCC
from torchfsdd import TorchFSDDGenerator, TrimSilence
from torchvision import datasets
from torchvision import transforms as vision_transforms


class FSDDMNISTDataset(multimodal_base.MultiModalDataset):
    """ The Spoken MNIST dataset. """

    def __init__(self,
                 phase,
                 audio_dataset,
                 mnist_dataset,
                 aligned_samples_index,
                 modality_sampler=None):
        super().__init__()

        self.phase = phase

        self.audio_dataset = audio_dataset
        self.mnist_dataset = mnist_dataset
        self.aligned_samples_index = aligned_samples_index

        self.modalities_name = ["audio", "rgb"]

        self.modality_sampler = modality_sampler

    def get_targets(self):
        """ Obtain the labels of samples in current phase dataset. """
        return self.mnist_dataset.targets

    def get_one_multimodal_sample(self, sample_idx):
        """ Obtain one multimodal sample from the dataset. """
        correspond_audio_index = self.aligned_samples_index[sample_idx]
        audio_sample, _ = self.audio_dataset[correspond_audio_index]
        rgb_sample, rgb_label = self.mnist_dataset[sample_idx]

        return {"audio": audio_sample, "rgb": rgb_sample}, rgb_label

    def __len__(self):
        return len(self.audio_dataset)


class DataSource(multimodal_base.MultiModalDataSource):
    """ The Spoken MNIST datasource ."""

    def __init__(self):
        super().__init__()
        _path = Config().params['data_path']

        self.data_name = Config().data.datasource
        self.modality_names = ["audio", "text"]

        # although we create train/test/val dirs for this dataset
        #   however, this dataset does not move actually data into these dirs
        #   but directly loading them from the 'recordings' when possible
        self._data_path_process(data_path=_path, base_data_name=self.data_name)

        # Create a transformation pipeline to apply to the recordings
        transforms = vision_transforms.Compose(
            [TrimSilence(threshold=1e-6),
             MFCC(sample_rate=8e3, n_mfcc=13)])

        # Fetch the latest version of FSDD and initialize a generator with those files
        fsdd = TorchFSDDGenerator(version='master',
                                  transforms=transforms,
                                  path=self.mm_data_info["base_data_dir_path"],
                                  load_all=True)

        # # Create a Torch dataset for the entire dataset from the generator
        # full_set = fsdd.full()
        # Create two Torch datasets for a train-test split from the generator
        self.audio_trainset, self.audio_testset = fsdd.train_test_split(
            test_size=0.1)
        self.audio_train_label = self.audio_trainset.labels
        self.audio_test_label = self.audio_testset.labels

        # # Create three Torch datasets for a train-validation-test split from the generator
        # train_set, val_set, test_set = fsdd.train_val_test_split(
        #     test_size=0.15, val_size=0.15)

        _transform = vision_transforms.Compose([
            vision_transforms.ToTensor(),
            vision_transforms.Normalize((0.1307, ), (0.3081, ))
        ])

        # get splits of mnist dataset
        self.mnist_trainset = datasets.MNIST(
            root=self.mm_data_info["base_data_dir_path"],
            train=True,
            download=True,
            transform=_transform)
        self.mnist_train_labels = self.mnist_trainset.train_labels
        self.mnist_testset = datasets.MNIST(
            root=self.mm_data_info["base_data_dir_path"],
            train=False,
            download=True,
            transform=_transform)
        self.mnist_test_labels = self.mnist_testset.test_labels

        self.train_multimodal_samples = self.align_audio_mnist(
            audio_label=self.audio_train_label,
            mnist_label=self.mnist_train_labels)

        self.test_multimodal_samples = self.align_audio_mnist(
            audio_label=self.audio_test_label,
            mnist_label=self.mnist_test_labels)

    def align_audio_mnist(self, audio_label, mnist_label):
        """ Align the audio and mnist in term of label. """
        aligned_samples_index = {}
        audio_label_index_pool = {
            label_id: np.where(np.array(audio_label) == label_id)[0]
            for label_id in list(set(audio_label))
        }
        audio_label_index_selected_count = {
            label_id: 0
            for label_id in list(set(audio_label))
        }
        mnist_label = mnist_label.tolist()
        for sample_index, sample_label in enumerate(mnist_label):

            total_label_index = audio_label_index_pool[sample_label]
            label_selected_count = audio_label_index_selected_count[
                sample_label]
            to_get_index_pos = label_selected_count % len(total_label_index)
            audio_sample_index = total_label_index[to_get_index_pos]

            aligned_samples_index[sample_index] = audio_sample_index

        return aligned_samples_index

    def get_train_set(self, modality_sampler=None):
        """ Obtain the train dataset. """
        train_set = FSDDMNISTDataset(
            phase="train",
            audio_dataset=self.audio_trainset,
            mnist_dataset=self.mnist_trainset,
            aligned_samples_index=self.train_multimodal_samples,
            modality_sampler=modality_sampler)
        return train_set

    def get_test_set(self, modality_sampler=None):
        """ Obtain the test dataset. """
        test_set = FSDDMNISTDataset(
            phase="train",
            audio_dataset=self.audio_testset,
            mnist_dataset=self.mnist_testset,
            aligned_samples_index=self.test_multimodal_samples,
            modality_sampler=modality_sampler)

        return test_set

    def num_train_examples(self):
        return 60000

    def num_test_examples(self):
        return 10000
