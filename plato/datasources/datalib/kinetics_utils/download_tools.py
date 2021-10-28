

import os
import json
import logging

from plato.datasources.datalib.kinetics_utils import parallel_downloader as parallel


def download_classes(splits_info, classes, num_workers, failed_save_file,
                     compress, verbose, skip, log_file):
    """ Download the specific classes """
    for list_path, save_root in splits_info:

        with open(list_path) as file:
            data = json.load(file)
        print("save_root: ", save_root)
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


def download_category(data_category_classes, category, num_workers,
                      failed_save_file, compress, verbose, skip, log_file):
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

    if category not in list(data_category_classes.keys()):
        raise ValueError("Category {} not found.".format(category))

    classes = data_category_classes[category]
    download_classes(classes, num_workers, failed_save_file, compress, verbose,
                     skip, log_file)


def download_train_val_sets(splits_info,
                            data_classes,
                            data_categories_file,
                            data_category_classes=None,
                            num_workers=4,
                            failed_log="train_val_failed_log.txt",
                            compress=False,
                            verbose=False,
                            skip=False,
                            log_file=None):
    """ Download all categories => all videos for train and the val set. """

    # # download the required categories in class-wise
    if os.path.exists(data_categories_file):
        with open(data_categories_file, "r") as file:
            categories = json.load(file)

        for category in categories:
            download_category(data_category_classes,
                              category,
                              num_workers,
                              failed_log,
                              compress=compress,
                              verbose=verbose,
                              skip=skip,
                              log_file=log_file)
    else:  # download all the classes in the training and val data files

        print("Downloading all classes directly")
        download_classes(splits_info, data_classes, num_workers, failed_log,
                         compress, verbose, skip, log_file)


def download_test_set(test_info_data_path, test_video_des_path, num_workers,
                      failed_log, compress, verbose, skip, log_file):
    """ Download the test set. """

    with open(test_info_data_path) as file:
        data = json.load(file)

    pool = parallel.VideoDownloaderPool(None,
                                        data,
                                        test_video_des_path,
                                        num_workers,
                                        failed_log,
                                        compress,
                                        verbose,
                                        skip,
                                        log_file=log_file)
    pool.start_workers()
    pool.feed_videos()
    pool.stop_workers()
