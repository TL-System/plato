"""
Base class for multimodal datasets.
"""

from abc import abstractmethod
import logging
import os
import subprocess
from collections import namedtuple

import torch
from torchvision.datasets.utils import download_url, extract_archive
from torchvision.datasets.utils import download_file_from_google_drive

from plato.datasources import base

TextData = namedtuple('TextData', ['caption', 'caption_phrases'])
BoxData = namedtuple('BoxData', ['caption_phrase_bboxs'])
TargetData = namedtuple('TargetData',
                        ['caption_phrases_cate', 'caption_phrases_cate_id'])


class MultiModalDataSource(base.DataSource):
    """
    The training or testing dataset that accommodates custom augmentation and transforms.
    """
    def __init__(self):
        super().__init__()

        # data name
        self.data_name = ""

        # the text name of the contained modalities
        self.modality_names = []

        # define the information container for the source data
        #  - source_data_path: the original downloaded data
        #  - base_data_dir_path: the source data used for the model
        # For some datasets, we directly utilize the base_data_dir_path as
        #  there is no need to process the original downloaded data to put them
        #  in the base_data_dir_path dir.
        self.mm_data_info = {"source_data_path": "", "base_data_dir_path": ""}

        # define the paths for the splited root data - train, test, and val
        self.splits_info = {
            "train": {
                "path": '',
                "split_anno_file": ''
            },
            "test": {
                "path": '',
                "split_anno_file": ''
            },
            "val": {
                "path": '',
                "split_anno_file": ''
            }
        }

    def set_modality_format(self, modality_name):
        """ An interface to set the modality name
            Thus, calling this func to obtain the modality name
             in all parts of the class to achieve the consistency
        """
        if modality_name in ["rgb", "flow"]:
            modality_format = "rawframes"
        else:  # convert to plurality
            modality_format = modality_name + "s"

        return modality_format

    def set_modality_path_key_format(self, modality_name):
        """ An interface to set the modality path
            Thus, calling this func to obtain the modality path
             in all parts of the class to achieve the consistency
        """
        modality_format = self.set_modality_format(modality_name)

        return modality_format + "_" + "path"

    def _create_modalities_path(self, modality_names=None):
        if modality_names is None:
            assert len(self.modality_names) != 0
            modality_names = self.modality_names

        for split_type in list(self.splits_info.keys()):
            split_path = self.splits_info[split_type]["path"]
            for modality_nm in modality_names:
                modality_format = self.set_modality_format(modality_nm)
                split_modality_path = os.path.join(split_path, modality_format)
                # modality data dir
                modality_path_format = self.set_modality_path_key_format(
                    modality_nm)
                self.splits_info[split_type][
                    modality_path_format] = split_modality_path
                if not os.path.exists(split_modality_path):
                    try:
                        os.makedirs(split_modality_path)
                    except FileExistsError:
                        pass

    def _data_path_process(
        self,
        data_path,  # the base directory for the data
        base_data_name=None):  # the directory name of the working data
        """ Generate the data structure based on the defined data path """

        # Create the full path by introducing the project path
        proj_root_path = os.path.abspath(os.curdir)
        base_data_path = os.path.join(proj_root_path, data_path,
                                      base_data_name)

        if not os.path.exists(base_data_path):
            os.makedirs(base_data_path)

        #
        self.mm_data_info["base_data_dir_path"] = base_data_path

        # create the split dirs for current dataset
        for split_type in list(self.splits_info.keys()):
            split_path = os.path.join(base_data_path, split_type)
            self.splits_info[split_type]["path"] = split_path
            if not os.path.exists(split_path):
                try:
                    os.makedirs(split_path)
                except FileExistsError:
                    pass

    def _download_arrange_data(
        self,
        download_url_address,
        put_data_dir,
        extract_to_dir=None,
        obtained_file_name=None,
    ):
        """ Download the raw data and arrange the data """
        # Extract to the same dir as the download dir
        if extract_to_dir is None:
            extract_to_dir = put_data_dir

        download_file_name = os.path.basename(download_url_address)
        download_file_path = os.path.join(put_data_dir, download_file_name)

        download_extracted_file_name = download_file_name.split(".")[0]
        download_extracted_dir_path = os.path.join(
            extract_to_dir, download_extracted_file_name)
        # Download the raw data if necessary
        if not self._exist_judgement(download_file_path):
            logging.info("Downloading the %s data.....", download_file_name)
            download_url(url=download_url_address,
                         root=put_data_dir,
                         filename=obtained_file_name)

        # Extract the data to the specific dir
        if ".zip" in download_file_name or ".tar.gz" in download_file_name:
            if not self._exist_judgement(download_extracted_dir_path):
                logging.info("Extracting data to %s dir.....", extract_to_dir)
                extract_archive(from_path=download_file_path,
                                to_path=extract_to_dir,
                                remove_finished=False)

        return download_extracted_file_name

    def _download_google_driver_arrange_data(
        self,
        download_file_id,
        extract_download_file_name,
        put_data_dir,
    ):
        download_data_file_name = extract_download_file_name + ".zip"
        download_data_path = os.path.join(put_data_dir,
                                          download_data_file_name)
        extract_data_path = os.path.join(put_data_dir,
                                         extract_download_file_name)
        if not self._exist_judgement(download_data_path):
            logging.info("Downloading the data to %s", download_data_path)
            download_file_from_google_drive(file_id=download_file_id,
                                            root=put_data_dir,
                                            filename=download_data_file_name)
        if not self._exist_judgement(extract_data_path):
            extract_archive(from_path=download_data_path,
                            to_path=put_data_dir,
                            remove_finished=True)

    def _exist_file_in_dir(self,
                           tg_file_name,
                           search_dir,
                           is_partial_name=True):
        """ Judge whether the input file exists in the search_dir. """
        # the tg_file_name matches one file if it match part of the file name
        if is_partial_name:
            is_included_fuc = lambda src_f_name: tg_file_name in src_f_name
        else:
            is_included_fuc = lambda src_f_name: tg_file_name == src_f_name
        is_existed = any(
            [is_included_fuc(f_name) for f_name in os.listdir(search_dir)])

        return is_existed

    def _exist_judgement(self, target_path):
        """ Judeg whether the input file/dir existed and whether it contains useful data """
        if not os.path.exists(target_path):
            logging.info("The path %s does not exist", target_path)
            return False

        # remove all .DS_Store files
        command = ['find', '.', '-name', '".DS_Store"', '-delete']
        command = ' '.join(command)
        #cmd = f"find . -name ".DS_Store" -delete"
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

        def get_size(folder):
            # get size
            size = 0
            for ele in os.scandir(folder):
                if not ele.name.startswith('.'):
                    size += os.path.getsize(ele)
            return size

        def is_contain_useful_file(target_dir):
            """ Return True once reaching one useful file """
            for _, _, files in os.walk(target_dir):
                for file in files:
                    # whether a useful file
                    if not file.startswith('.'):
                        return True
            return False

        if os.path.isdir(target_path):
            if get_size(target_path
                        ) == 0 or not is_contain_useful_file(target_path):
                logging.info("The path %s does exist but contains no data",
                             target_path)
                return False
            else:
                return True

        logging.info("The file %s does exist", target_path)
        return True

    def num_modalities(self) -> int:
        """ Number of modalities """
        return len(self.modality_names)

    @abstractmethod
    def get_phase_dataset(self, phase, modality_sampler):
        """ Obtain the dataset with the modaltiy_sampler for the
            specific phase (train/test/val) """
        raise NotImplementedError(
            "Please Implement 'get_phase_dataset' method")

    @abstractmethod
    def get_train_set(self, modality_sampler):
        """ Obtain the train dataset with the modaltiy_sampler """
        raise NotImplementedError("Please Implement 'get_train_set' method")

    @abstractmethod
    def get_test_set(self, modality_sampler):
        """ btain the test dataset with the modaltiy_sampler """
        raise NotImplementedError("Please Implement 'get_test_set' method")


class MultiModalDataset(torch.utils.data.Dataset):
    """ The base interface for the multimodal data """
    def __init__(self):
        self.phase = None  # the 'train' , 'test', 'val'

        # the recorded samples for current dataset
        #   In flickr20K entities dataset, this presents as:
        #    this is a dict in which key is the 'sample name/id' ...
        #    the values are the sample's information,
        #    for example: the annotation with its bounding boxes ...
        #   In the Kinetics, this presents as:
        #    this is a dict:
        #    {"rgb": rgb_dataset, "flow": flow_dataset, "audio": audio_dataset}
        self.phase_multimodal_data_record = None

        # the detailed info in selected split
        #   i.e., path, path for different modalities, et. al
        self.phase_info = None
        # the data types included,
        #  e.g. in flickr30k entities, ["Images", "Annotations", "Sentences"]
        self.data_types = None

        # the name of the modalities in the dataset
        self.modalities_name = None

        # the sampler for modalities,
        #   specific modalities can be masked by this sampler
        self.modality_sampler = None
        # transformation func for image and text if provided
        self.transform_image_dec_func = None
        self.transform_text_func = None

        # the basic modalities
        self.basic_modalities = ["rgb", "flow", "text", "audio"]
        # the additional data/annotations
        self.basic_items = ["box", "target"]

    @abstractmethod
    def get_targets(self):
        """ Obtain the labels of samples in current phase dataset. """
        raise NotImplementedError("Please Implement the 'targets' function")

    @abstractmethod
    def get_one_multimodal_sample(self, sample_idx):
        """ Get the sample containing different modalities.
            Different multi-modal datasets should have their
             personal 'get_one_multimodal_sample' method.


            Args:
                sample_idx (int): the index of the sample

            Output:
                a dict containing different modalities, the
                 key of the dict is the modality name that should
                 be included in the basic_modalities and basic_items.
         """
        raise NotImplementedError(
            "Please Implement the 'get_one_multimodal_sample(self, sample_idx)' function"
        )

    def __getitem__(self, sample_idx):
        """Get the sample for either training or testing given index."""
        sampled_multimodal_data = self.get_one_multimodal_sample(sample_idx)

        # utilize the modality to mask specific modalities
        sampled_modality_data = dict()
        for item_name, item_data in sampled_multimodal_data.items():
            # maintain the modality data based on the sampler
            # maintain the external data
            if item_name in self.modality_sampler or \
                item_name in self.basic_items:
                sampled_modality_data[item_name] = item_data

        return sampled_modality_data

    @abstractmethod
    def __len__(self):
        """ obtain the length of the multi-modal data """
        raise NotImplementedError("Please Implement this method")
