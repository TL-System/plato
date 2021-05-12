"""
The functions for downloading the datasets in parallel.
"""

import os
from multiprocessing import Process, Queue

from plato.datasources.datalib import downloader


class VideoDownloaderPool:
    """ A pool of video downloaders. """
    def __init__(
            self,
            classes,  # List of classes to download.
            videos_dict,  # Dictionary of all videos.
            directory,  # Where to download to videos.
            num_workers,  # How many videos to download in parallel.
            failed_save_file,  # Where to save the failed videos ids.
            compress,  # Whether to compress the videos using gzip.
            verbose,
            skip,
            log_file=None):

        self.classes = classes
        self.videos_dict = videos_dict
        self.directory = directory
        self.num_workers = num_workers
        self.failed_save_file = failed_save_file
        self.compress = compress
        self.verbose = verbose
        self.skip = skip
        self.log_file = log_file

        self.videos_queue = Queue(100)
        self.failed_queue = Queue(100)

        self.workers = []
        self.failed_save_worker = None

        if verbose:
            print("downloading:")
            if self.classes is not None:
                for cls in self.classes:
                    print(cls)
                print()

    def feed_videos(self):
        """ Feed video ids into the download queue. """
        if self.classes is None:
            downloader.download_class_parallel(None, self.videos_dict,
                                               self.directory,
                                               self.videos_queue)
        else:
            for class_name in self.classes:

                if self.verbose:
                    print(class_name)

                class_path = os.path.join(self.directory,
                                          class_name.replace(" ", "_"))

                if not self.skip or not os.path.isdir(class_path):
                    downloader.download_class_parallel(class_name,
                                                       self.videos_dict,
                                                       self.directory,
                                                       self.videos_queue)

            if self.verbose:
                print("done")

    def start_workers(self):
        """ Start all workers. """

        # start failed videos saver
        if self.failed_save_file is not None:
            self.failed_save_worker = Process(target=write_failed_worker,
                                              args=(self.failed_queue,
                                                    self.failed_save_file))
            self.failed_save_worker.start()

        # start download workers
        for _ in range(self.num_workers):
            worker = Process(target=video_worker,
                             args=(self.videos_queue, self.failed_queue,
                                   self.compress, self.log_file))
            worker.start()
            self.workers.append(worker)

    def stop_workers(self):
        """ Stop all workers. """

        # send end signal to all download workers
        for _ in range(len(self.workers)):
            self.videos_queue.put(None)

        # wait for the processes to finish
        for worker in self.workers:
            worker.join()

        # end failed videos saver
        if self.failed_save_worker is not None:
            self.failed_queue.put(None)
            self.failed_save_worker.join()


def video_worker(videos_queue, failed_queue, compress, log_file):
    """[Downloads videos pass in the videos queue.]

    Args:
        videos_queue ([Queue]): [Queue for metadata of videos to be download.]
        failed_queue ([Queue]): [Queue of failed video ids.]
        compress ([Boolean]): [Whether to compress the videos using gzip.]
        log_file ([type]): [Path to a log file for youtube-dl.]
    """
    while True:
        request = videos_queue.get()

        if request is None:
            break

        video_id, directory, start, end = request

        if not downloader.process_video(video_id,
                                        directory,
                                        start,
                                        end,
                                        compress=compress,
                                        log_file=log_file):
            failed_queue.put(video_id)


def write_failed_worker(failed_queue, failed_save_file):
    """[Write failed video ids into a file.]

    Args:
        failed_queue ([Queue]): [Queue of failed video ids.]
        failed_save_file ([Queue]): [Where to save the videos.]
    """
    file = open(failed_save_file, "a")

    while True:
        video_id = failed_queue.get()

        if video_id is None:
            break

        file.write("{}\n".format(video_id))

    file.close()
