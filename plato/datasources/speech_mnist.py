"""
This multi-modal dataset, spoken_mnist, derives from 
https://zenodo.org/record/3515935#collapseTwo.

The datasets consists of 70000 images (60000 for training and 10000 for test) of 28 x 28 = 784 dimensions.

- Audio part:
 The spoken digits database was extracted from an audio dataset Google Speech Commands. 
 It consists of 105829 utterances of 35 words, amongst which 38908 utterances of the ten digits 
 -- 34801 for training 
 -- 4107 for test.
 A pre-processing was done via the extraction of the Mel Frequency Cepstral Coefficients (MFCC) with a 
 framing window size of 50 ms and frame shift size of 25 ms. Since the speech samples are approximately 
 1 s long, we end up with 39 time slots. For each one, we extract 12 MFCC coefficients with an additional 
 energy coefficient. Thus, we have a final vector of 39 x 13 = 507 dimensions. Standardization and 
 normalization were applied on the MFCC features.

- RGB part:
 The original MNIST handwritten digits database with no additional processing.
 It consists of 70000 images (60000 for training and 10000 for test) of 28 x 28 = 784 dimensions.

-> To construct the multimodal digits dataset, written and spoken digits of the same class respecting the above initial partitioning are associated. Since there are less samples for the spoken digits, we duplicated some random samples to match the number of written digits and have a multimodal digits database of 70000 samples (60000 for training and 10000 for test).

"""

import os

import numpy as np

from plato.config import Config
from plato.datasources import multimodal_base


class SPMNISTDataset(multimodal_base.MultiModalDataset):
    """ The Spoken MNIST dataset. """

    def __init__(self, phase, data_dir, modality_sampler=None):
        super().__init__()

        self.phase = phase

        self.modalities_name = ["audio", "rgb"]

        self.modality_sampler = modality_sampler

        basic_audio_file_name = "data_sp_{}.npy"
        basic_rgb_file_name = "data_wr_{}.npy"
        basic_label_file_name = "labels_{}.npy"
        audio_samples_path = os.path.join(
            data_dir, basic_audio_file_name.format(self.phase))
        rgb_samples_path = os.path.join(data_dir,
                                        basic_rgb_file_name.format(self.phase))
        samples_label_path = os.path.join(
            data_dir, basic_label_file_name.format(self.phase))

        self.phase_info = {
            "audio": audio_samples_path,
            "rgb": rgb_samples_path
        }

        self.audio_data = np.load(audio_samples_path)
        self.rgb_data = np.load(rgb_samples_path)
        self.label_data = np.load(samples_label_path)

    def get_targets(self):
        """ Obtain the labels of samples in current phase dataset. """
        return self.label_data

    def get_one_multimodal_sample(self, sample_idx):
        """ Obtain one multimodal sample from the dataset. """
        audio_sample = self.audio_data[sample_idx]
        rgb_sample = self.rgb_data[sample_idx]
        sample_label = self.label_data[sample_idx]

        return {"audio": audio_sample, "rgb": rgb_sample}, sample_label

    def __len__(self):
        return len(self.audio_data)


class DataSource(multimodal_base.MultiModalDataSource):
    """The Spoken MNIST datasource ."""

    def __init__(self):
        super().__init__()
        _path = Config().data.data_path

        self.data_name = Config().data.datasource
        self.modality_names = ["audio", "text"]

        self._data_path_process(data_path=_path, base_data_name=self.data_name)

        base_download_url = "https://zenodo.org/record/3515935/files/{}.npy?download=1"

        ## train, 60000 samples:
        files = {
            "train": ("data_sp_train", "data_sp_train", "labels_train"),
            "test": ("data_wr_test", "data_sp_test", "labels_test")
        }
        for split_name in list(files.keys()):
            split_files_name = files[split_name]
            split_dir = self.splits_info[split_name]["path"]
            for file_name in split_files_name:
                file_url = base_download_url.format(file_name)

                self._download_arrange_data(
                    download_url_address=file_url,
                    put_data_dir=split_dir,
                    obtained_file_name=file_name + ".npy",
                )

    def get_train_set(self, modality_sampler=None):
        """ Obtain the train dataset. """
        train_set = SPMNISTDataset(phase="train",
                                   data_dir=self.splits_info["train"]["path"],
                                   modality_sampler=modality_sampler)

        return train_set

    def get_test_set(self, modality_sampler=None):
        """ Obtain the test dataset. """
        test_set = SPMNISTDataset(phase="test",
                                  data_dir=self.splits_info["test"]["path"],
                                  modality_sampler=modality_sampler)

        return test_set

    def num_train_examples(self):
        return 60000

    def num_test_examples(self):
        return 10000
