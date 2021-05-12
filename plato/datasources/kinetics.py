"""
The Kinetics700 dataset.

Note that the setting for the data loader is obtained from the github repo provided by the official workers:
https://github.com/pytorch/vision/references/video_classification/train.py
"""

import json
import logging
import os
import sys

from torch.utils.data.dataloader import default_collate
from torchvision import datasets

from plato.config import Config
from plato.datasources import base
from plato.datasources.datalib import parallel_downloader as parallel
from plato.datasources.datalib import video_transform


class DataSource(base.DataSource):
    """The Kinetics700 dataset."""
    def __init__(self):
        super().__init__()
        _path = Config().data.data_path
        if not os.path.exists(_path):
            os.makedirs(_path)

        download_url = Config().data.download_url
        if not os.path.exists(_path + download_url.split('/')[-1]):
            logging.info(
                "Downloading the Kinetics700 dataset. This may take a while.")
            DataSource.download(download_url, _path)

        self.source_data_path = _path
        meta_dirs = [
            meta_dir for meta_dir in os.listdir(_path)
            if "kinetics" in meta_dir
        ]
        # obtain the path of the data information
        tg_meta_dir = meta_dirs[0]
        self.tg_meta_categories = os.path.join(self.source_data_path,
                                               "categories.json")
        self.tg_meta_classes = os.path.join(self.source_data_path,
                                            "classes.json")
        self.train_meta_data_path = os.path.join(self.source_data_path,
                                                 tg_meta_dir, "train.json")
        self.test_meta_data_path = os.path.join(self.source_data_path,
                                                tg_meta_dir, "test.json")
        self.val_meta_data_path = os.path.join(self.source_data_path,
                                               tg_meta_dir, "validate.json")

        # define the paths for the splited data - train, test, and val
        self.train_root_path = os.path.join(_path, "train")
        self.test_root_path = os.path.join(_path, "test")
        self.val_root_path = os.path.join(_path, "val")

        self.data_classes = self.classes()

        # get the download hyper-parameters
        num_workers = Config().data.num_workers
        failed_save_file = Config().data.failed_save_file
        compress = Config().data.compress
        verbose = Config().data.verbose
        skip = Config().data.skip
        log_file = Config().data.log_file

        failed_save_file = os.path.join(self.source_data_path,
                                        failed_save_file)

        # download the raw dataset if necessary
        if not os.path.exists(self.train_root_path):
            logging.info(
                "Downloading the raw videos for the Kinetics700 dataset. This may take a long time."
            )
            self.maybe_create_dirs()

            self.download_train_val_sets(num_workers=num_workers,
                                         failed_log=failed_save_file,
                                         compress=compress,
                                         verbose=verbose,
                                         skip=skip,
                                         log_file=os.path.join(
                                             self.source_data_path, log_file))

            self.download_test_set(num_workers=num_workers,
                                   failed_log=failed_save_file,
                                   compress=compress,
                                   verbose=verbose,
                                   skip=skip,
                                   log_file=os.path.join(
                                       self.source_data_path, log_file))

        # obtain the data loader settings
        self.clip_len = Config().data.clip_len
        self.clips_per_video = Config().data.clips_per_video

    def maybe_create_dirs(self):
        # Create directories for training, validation and testing videos if they do not exist.
        for path in [
                self.train_root_path, self.test_root_path, self.val_root_path
        ]:
            if not os.path.exists(path):
                try:
                    os.makedirs(path)
                except FileExistsError:
                    pass

    def download_category(self, category, num_workers, failed_save_file,
                          compress, verbose, skip, log_file):
        """[Download all videos that belong to the given category.]

        Args:
            category ([str]): [The category to download.]
            num_workers ([int]): [Number of downloads in parallel.]
            failed_save_file ([str]): [Where to save failed video ids.]
            compress ([bool]): [Decides if the videos should be compressed.]
            verbose ([bool]): [Print status.]
            skip ([bool]): [Skip classes that already have folders (i.e. at least one video was downloaded).]
            log_file ([str]): [Path to log file for youtube-dl.]

        Raises:
            ValueError: [description]
        """
        if os.path.exists(self.tg_meta_categories):
            with open(self.tg_meta_categories, "r") as file:
                categories = json.load(file)

            if category not in categories:
                raise ValueError("Category {} not found.".format(category))

        classes = categories[category]
        self.download_classes(classes, num_workers, failed_save_file, compress,
                              verbose, skip, log_file)

    def download_classes(self, classes, num_workers, failed_save_file,
                         compress, verbose, skip, log_file):
        """ Download the specific classes """
        for list_path, save_root in zip(
            [self.train_meta_data_path, self.val_meta_data_path],
            [self.train_root_path, self.val_root_path]):
            with open(list_path) as file:
                data = json.load(file)

            pool = parallel.VideoDownloaderPool(classes,
                                                data,
                                                save_root,
                                                num_workers,
                                                failed_save_file,
                                                compress,
                                                verbose,
                                                skip,
                                                log_file=log_file)
            pool.start_workers()
            pool.feed_videos()
            pool.stop_workers()

    def download_train_val_sets(self,
                                num_workers=4,
                                failed_log="train_val_failed_log.txt",
                                compress=False,
                                verbose=False,
                                skip=False,
                                log_file=None):
        """ Download all categories => all videos for train and the val set. """

        # # download the required categories in class-wise
        if os.path.exists(self.tg_meta_categories):
            with open(self.tg_meta_categories, "r") as file:
                categories = json.load(file)

            for category in categories:
                self.download_category(category,
                                       num_workers,
                                       failed_log,
                                       compress=compress,
                                       verbose=verbose,
                                       skip=skip,
                                       log_file=log_file)
        else:  # download all the classes in the training and val data files

            self.download_classes(self.data_classes, num_workers, failed_log,
                                  compress, verbose, skip, log_file)

    def download_test_set(self, num_workers, failed_log, compress, verbose,
                          skip, log_file):
        """ Download the test set. """

        with open(self.test_meta_data_path) as file:
            data = json.load(file)

        pool = parallel.VideoDownloaderPool(None,
                                            data,
                                            self.test_root_path,
                                            num_workers,
                                            failed_log,
                                            compress,
                                            verbose,
                                            skip,
                                            log_file=log_file)
        pool.start_workers()
        pool.feed_videos()
        pool.stop_workers()

    def classes(self):
        """ Obtain a list of class names in the dataset. """

        classes_container = list()
        if os.path.exists(self.tg_meta_classes):
            with open(self.tg_meta_classes, "r") as class_file:
                lines = class_file.readlines()
                classes_container = [line.replace("\n", "") for line in lines]

            return classes_container

        if not os.path.exists(self.train_meta_data_path) or not os.path.exists(
                self.val_meta_data_path):
            logging.info(
                "The json files of the dataset are not completed. Download it first."
            )
            sys.exit()

        for list_path in [self.train_meta_data_path, self.val_meta_data_path]:
            with open(list_path) as file:
                videos_data = json.load(file)
            for key in videos_data.keys():
                metadata = videos_data[key]
                annotations = metadata["annotations"]
                label = annotations["label"]
                class_name = label.replace("_", " ")
                if class_name not in classes_container:
                    classes_container.append(class_name)
        with open(self.tg_meta_classes, "w") as file:
            for class_name in classes_container:
                file.write(class_name)
                file.write('\n')

        return classes_container

    def num_train_examples(self):
        if not os.path.exists(self.train_root_path):
            return 0
        return len(os.listdir(self.train_root_path))

    def num_test_examples(self):
        if not os.path.exists(self.test_root_path):
            return 0
        return len(os.listdir(self.test_root_path))

    def get_train_set(self):
        transform_train = video_transform.VideoClassificationTrainTransformer(
            (128, 171), (112, 112))
        kinetics_train_data = datasets.Kinetics400(
            root=self.train_root_path,
            frames_per_clip=self.clip_len,
            step_between_clips=1,
            transform=transform_train,
            frame_rate=15,
            extensions=(
                'avi',
                'mp4',
            ))
        return kinetics_train_data

    def get_val_set(self):
        transform_val = video_transform.VideoClassificationEvalTransformer(
            (128, 171), (112, 112))
        kinetics_val_data = datasets.Kinetics400(root=self.val_root_path,
                                                 frames_per_clip=self.clip_len,
                                                 step_between_clips=1,
                                                 transform=transform_val,
                                                 frame_rate=15,
                                                 extensions=(
                                                     'avi',
                                                     'mp4',
                                                 ))
        return kinetics_val_data

    def get_test_set(self):
        transform_test = video_transform.VideoClassificationEvalTransformer(
            (128, 171), (112, 112))
        kinetics_test_data = datasets.Kinetics400(
            root=self.test_root_path,
            frames_per_clip=self.clip_len,
            step_between_clips=1,
            transform=transform_test,
            frame_rate=15,
            extensions=(
                'avi',
                'mp4',
            ))

    @staticmethod
    def get_data_loader(self, batch_size, dataset, sampler):
        def collate_fn(batch):
            return default_collate

        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           sampler=sampler,
                                           pin_memory=True,
                                           collate_fn=collate_fn)
