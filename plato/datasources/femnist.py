"""
The FEMNIST Classification dataset.

FEMNIST contains 817851 images, each of which is a 28x28 greyscale image in 1 out of 62 classes.
The dataset is already partitioned by clients' identification.
There are in total 3597 clients, each of which has 227.37 images on average (std is 88.84).
For each client, 90% data samples are used for training, while the rest are used for testing.

Reference for the dataset: Cohen, G., Afshar, S., Tapson, J. and Van Schaik, A.,
EMNIST: Extending MNIST to handwritten letters. In 2017 IEEE IJCNN.
Reference for the related submodule: https://github.com/TalwalkarLab/leaf/tree/master
"""

from __future__ import division
import logging
import os

from torchvision import transforms

from plato.config import Config
from plato.datasources import base

import json
import shutil
import random
import pickle
import hashlib
import numpy as np
from PIL import Image
from collections import defaultdict, OrderedDict
from zhifeng_learning.patch.utils import (
    CustomDictDataset, ReshapeListTransform
)


class DataSource(base.DataSource):
    """The FEMNIST dataset."""
    def __init__(self):
        super().__init__()
        self.trainset_size = 0
        self.testset_size = 0

        data_path = os.path.join(Config().data.data_path, 'FEMNIST')
        train_dir = os.path.join(data_path, 'ready', 'train')
        test_dir = os.path.join(data_path, 'ready', 'test')
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            self.preprocess_data(data_path=data_path)

        logging.info("Loading the FEMNIST dataset. This may take a while.")
        train_clients, _, train_data, test_data = self.read_data(train_dir, test_dir)
        trainset = self.dict_to_list(train_clients, train_data)
        testset = self.merge_testset(train_clients, test_data)

        _transform = transforms.Compose([
            ReshapeListTransform((28, 28, 1)),
            transforms.ToPILImage(),
            transforms.RandomCrop(28, padding=2, padding_mode="constant", fill=1.0),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2), ratio=(4. / 5., 5. / 4.)),
            transforms.RandomRotation(5, fill=1.0),
            transforms.ToTensor(),
            transforms.Normalize(0.9637, 0.1597),
        ])
        self.trainset = [CustomDictDataset(dictionary=d, transform=_transform) for d in trainset]
        self.testset = CustomDictDataset(dictionary=testset, transform=_transform)

    def dict_to_list(self, list_of_keys, dictionary):
        result = []
        for key in list_of_keys:
            result.append(dictionary[key])

        return result

    def merge_testset(self, list_of_keys, dictionary):
        first_key = list_of_keys[0]
        result = dictionary[first_key]
        for key in list_of_keys[1:]:
            result['x'].extend(dictionary[key]['x'])
            result['y'].extend(dictionary[key]['y'])

        self.testset_size = len(result['x'])
        return result

    def do_nothing(self):
        pass

    def read_dir_worker(self, files, data_dir):
        clients = []
        groups = []
        data = defaultdict(self.do_nothing)

        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            data.update(cdata['user_data'])

        clients = list(sorted(data.keys()))
        return clients, groups, data

    def read_dir(self, data_dir):
        clients = []
        groups = []
        data = defaultdict(lambda: None)

        files = os.listdir(data_dir)
        files = [f for f in files if f.endswith('.json')]

        # no multiprocessing due to memory concerns
        for f in files:
            file_path = os.path.join(data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            clients.extend(cdata['users'])
            if 'hierarchies' in cdata:
                groups.extend(cdata['hierarchies'])
            data.update(cdata['user_data'])

        clients = list(sorted(data.keys()))
        return clients, groups, data

    def read_data(self, train_data_dir, test_data_dir):
        train_clients, train_groups, train_data = self.read_dir(train_data_dir)
        test_clients, test_groups, test_data = self.read_dir(test_data_dir)

        assert train_clients == test_clients
        assert train_groups == test_groups

        return train_clients, train_groups, train_data, test_data

    def get_class_files(self, raw_dir):
        class_files = []  # (class, file directory)

        # directory hierarchy: by_class -> classes
        # -> folders containing images -> images
        class_dir = os.path.join(raw_dir, 'by_class')
        classes = os.listdir(class_dir)
        classes = [c for c in classes if len(c) == 2]

        for cl in classes:
            cldir = os.path.join(class_dir, cl)
            subcls = os.listdir(cldir)
            subcls = [s for s in subcls if (('hsf' in s) and ('mit' not in s))]

            for subcl in subcls:
                subclddir = os.path.join(cldir, subcl)
                images = os.listdir(subclddir)
                images_dirs = [os.path.join(subclddir, i) for i in images]

                for image_dir in images_dirs:
                    class_files.append((cl, image_dir))

        return class_files

    def get_write_files(self, raw_dir):
        write_files = []   # (writer, file directory)

        # directory hierarchy: by_write -> folders containing writers
        # -> writer -> types of images -> images
        write_dir = os.path.join(raw_dir, 'by_write')
        write_parts = os.listdir(write_dir)

        for write_part in write_parts:
            writers_dir = os.path.join(write_dir, write_part)
            writers = os.listdir(writers_dir)

            for writer in writers:
                writer_dir = os.path.join(writers_dir, writer)
                wtypes = os.listdir(writer_dir)

                for wtype in wtypes:
                    type_dir = os.path.join(writer_dir, wtype)
                    rel_type_dir = os.path.join(writer_dir, wtype)
                    images = os.listdir(type_dir)
                    image_dirs = [os.path.join(rel_type_dir, i) for i in images]

                    for image_dir in image_dirs:
                        write_files.append((writer, image_dir))

        return write_files

    def extract_class_file_dirs(self, raw_dir, class_file_dirs_path):
        logging.info("Extracting meta info of images in the `by_class` folder.")
        class_files = self.get_class_files(raw_dir)  # (class, file directory)
        with open(class_file_dirs_path, 'wb') as fout:
            pickle.dump(class_files, fout, pickle.HIGHEST_PROTOCOL)
        logging.info("Extracted.")

    def extract_write_file_dirs(self, raw_dir, write_file_dirs_path):
        logging.info("Extracting meta info of images in the `by_write` folder.")
        write_files = self.get_write_files(raw_dir)  # (writer, file directory)
        with open(write_file_dirs_path, 'wb') as fout:
            pickle.dump(write_files, fout, pickle.HIGHEST_PROTOCOL)
        logging.info("Extracted.")

    def compute_file_hashes(self, file_dirs_path, file_hashes_path, related_folder):
        logging.info(f"Computing hashes for images in the `{related_folder}` folder.")
        with open(file_dirs_path, 'rb') as fin:
            file_dirs = pickle.load(fin)

        file_hashes = []
        for tup in file_dirs:
            (cclass, cfile) = tup
            chash = hashlib.md5(open(cfile, 'rb').read()).hexdigest()
            file_hashes.append((cclass, cfile, chash))

        with open(file_hashes_path, 'wb') as fout:
            pickle.dump(file_hashes, fout, pickle.HIGHEST_PROTOCOL)
        logging.info("Computed.")

    def match_hashes(self, class_file_hashes_path, write_file_hashes_path,
                     write_with_class_path):
        logging.info("Matching hashes so that class labels"
                     " can be assigned to images in the `by_write` folder")
        with open(class_file_hashes_path, 'rb') as fin:
            class_file_hashes = pickle.load(fin)
        with open(write_file_hashes_path, 'rb') as fin:
            write_file_hashes = pickle.load(fin)

        class_hash_dict = {}
        for i in range(len(class_file_hashes)):
            (c, f, h) = class_file_hashes[len(class_file_hashes) - i - 1]
            class_hash_dict[h] = (c, f)

        write_classes = []
        for tup in write_file_hashes:
            (w, f, h) = tup
            write_classes.append((w, f, class_hash_dict[h][0]))

        with open(write_with_class_path, 'wb') as fout:
            pickle.dump(write_classes, fout, pickle.HIGHEST_PROTOCOL)
        logging.info("Matched.")

    def group_by_writers(self, write_with_class_path, images_by_writer_path):
        logging.info("Grouping images by writers.")
        with open(write_with_class_path, 'rb') as fin:
            write_classes = pickle.load(fin)

        writers = []
        cimages = []
        (cw, _, _) = write_classes[0]
        for (w, f, c) in write_classes:
            if not w == cw:
                writers.append((cw, cimages))
                cw = w
                cimages = [(f, c)]
            cimages.append((f, c))
        writers.append((cw, cimages))

        with open(images_by_writer_path, 'wb') as fout:
            pickle.dump(writers, fout, pickle.HIGHEST_PROTOCOL)
        logging.info("Grouped.")

    def relabel_class(self, c):
        if c.isdigit() and int(c) < 40:
            return int(c) - 30
        elif int(c, 16) <= 90:  # upper case
            return int(c, 16) - 55
        else:
            return int(c, 16) - 61

    def summarize_all_data(self, images_by_writer_path, all_data_dir):
        logging.info("Summarizing data to the `all_data` folder.")

        os.makedirs(all_data_dir)
        max_writers = 100  # maximum number of writers per json file
        with open(images_by_writer_path, 'rb') as fin:
            writers = pickle.load(fin)

        users = []
        num_samples = []
        user_data = {}
        writer_count, all_writers = 0, 0
        json_index = 0

        for (w, l) in writers:
            users.append(w)
            num_samples.append(len(l))
            user_data[w] = {'x': [], 'y': []}

            size = 28, 28  # original size is 128, 128
            for (f, c) in l:
                image = Image.open(f)
                gray = image.convert('L')
                gray.thumbnail(size, Image.ANTIALIAS)
                arr = np.asarray(gray).copy()
                vec = arr.flatten()
                vec = vec / 255  # scale all pixel values to [0, 1]
                vec = vec.tolist()

                nc = self.relabel_class(c)

                user_data[w]['x'].append(vec)
                user_data[w]['y'].append(nc)

            writer_count += 1
            all_writers += 1

            if writer_count == max_writers or all_writers == len(writers):
                all_data = {
                    'users': users,
                    'num_samples': num_samples,
                    'user_data': user_data,
                }

                file_name = 'all_data_%d.json' % json_index
                file_path = os.path.join(all_data_dir, file_name)
                with open(file_path, 'w') as fout:
                    json.dump(all_data, fout)

                writer_count = 0
                json_index += 1
                users[:] = []
                num_samples[:] = []
                user_data.clear()

        logging.info("Summarized.")

    def sample_data(self, all_data_dir, sampled_data_dir, scale, rng_seed):
        # Equivalent to executing `LEAF_DATA_META_DIR=meta python3 sample.py --name femnist
        # --fraction [scale] --seed [rng_seed]` at ~/data/utils/ of LEAF
        logging.info(f"Sampling {scale} data using seed {rng_seed}")

        os.makedirs(sampled_data_dir)
        files = os.listdir(all_data_dir)
        files = [f for f in files if f.endswith('.json')]
        rng = random.Random(rng_seed)

        for f in files:
            file_dir = os.path.join(all_data_dir, f)
            with open(file_dir, 'r') as fin:
                # Load data into an OrderedDict, to prevent ordering changes
                # and enable reproducibility
                data = json.load(fin, object_pairs_hook=OrderedDict)

            tot_num_samples = sum(data['num_samples'])
            num_new_samples = int(scale * tot_num_samples)
            hierarchies = None

            ctot_num_samples = 0
            users = data['users']
            users_and_hiers = None

            if 'hierarchies' in data:
                users_and_hiers = list(zip(users, data['hierarchies']))
                rng.shuffle(users_and_hiers)
            else:
                rng.shuffle(users)

            user_i = 0
            num_samples = []
            user_data = {}

            if 'hierarchies' in data:
                hierarchies = []

            while ctot_num_samples < num_new_samples:
                if users_and_hiers is not None:
                    user, hier = users_and_hiers[user_i]
                else:
                    user = users[user_i]

                cdata = data['user_data'][user]
                cnum_samples = len(data['user_data'][user]['y'])

                if ctot_num_samples + cnum_samples > num_new_samples:
                    cnum_samples = num_new_samples - ctot_num_samples
                    indices = [i for i in range(cnum_samples)]
                    new_indices = rng.sample(indices, cnum_samples)

                    x, y = [], []
                    for i in new_indices:
                        x.append(data['user_data'][user]['x'][i])
                        y.append(data['user_data'][user]['y'][i])
                    cdata = {'x': x, 'y': y}

                if 'hierarchies' in data:
                    hierarchies.append(hier)

                num_samples.append(cnum_samples)
                user_data[user] = cdata
                ctot_num_samples += cnum_samples
                user_i += 1

            if 'hierarchies' in data:
                users = [u for u, h in users_and_hiers][:user_i]
            else:
                users = users[:user_i]

            all_data = {
                'users': users,
                'num_samples': num_samples,
                'user_data': user_data,
            }
            if hierarchies is not None:
                all_data['hierarchies'] = hierarchies

            label_str = 'niid'
            frac_str = str(scale)[2:]
            file_name = '%s_%s_%s.json' % ((f[:-5]), label_str, frac_str)
            out_dir = os.path.join(sampled_data_dir, file_name)
            with open(out_dir, 'w') as fout:
                json.dump(all_data, fout)

        logging.info(f"Sampled.")

    def split_data(self, sampled_data_dir, train_dir, test_dir, train_fraction, rng_seed):
        # Equivalent to executing `LEAF_DATA_META_DIR=meta python3 split_data.py --by_sample
        # --name femnist --frac [train_fraction] --seed [rng_seed]` at ~/data/utils/ of LEAF
        logging.info(f"Splitting datasets for training and testing using seed {rng_seed}")

        os.makedirs(train_dir)
        os.makedirs(test_dir)
        files = os.listdir(sampled_data_dir)
        files = [f for f in files if f.endswith('.json')]
        rng = random.Random(rng_seed)

        arg_label = str(train_fraction)[2:]

        # check if data contains information on hierarchies
        file_dir = os.path.join(sampled_data_dir, files[0])
        with open(file_dir, 'r') as fin:
            data = json.load(fin)
        include_hierarchy = 'hierarchies' in data

        for f in files:
            file_dir = os.path.join(sampled_data_dir, f)
            with open(file_dir, 'r') as fin:
                # Load data into an OrderedDict, to prevent ordering changes
                # and enable reproducibility
                data = json.load(fin, object_pairs_hook=OrderedDict)

            num_samples_train = []
            user_data_train = {}
            num_samples_test = []
            user_data_test = {}
            user_indices = []  # indices of users in data['users'] that are not deleted

            for i, u in enumerate(data['users']):
                curr_num_samples = len(data['user_data'][u]['y'])
                if curr_num_samples >= 2:
                    # ensures number of train and test samples both >= 1
                    num_train_samples = max(1, int(train_fraction * curr_num_samples))
                    if curr_num_samples == 2:
                        num_train_samples = 1

                    num_test_samples = curr_num_samples - num_train_samples

                    indices = [j for j in range(curr_num_samples)]
                    train_indices = rng.sample(indices, num_train_samples)
                    test_indices = [i for i in range(curr_num_samples) if i not in train_indices]

                    if len(train_indices) >= 1 and len(test_indices) >= 1:
                        user_indices.append(i)
                        num_samples_train.append(num_train_samples)
                        num_samples_test.append(num_test_samples)
                        user_data_train[u] = {'x': [], 'y': []}
                        user_data_test[u] = {'x': [], 'y': []}

                        train_blist = [False for _ in range(curr_num_samples)]
                        test_blist = [False for _ in range(curr_num_samples)]

                        for j in train_indices:
                            train_blist[j] = True
                        for j in test_indices:
                            test_blist[j] = True

                        for j in range(curr_num_samples):
                            if train_blist[j]:
                                user_data_train[u]['x'].append(data['user_data'][u]['x'][j])
                                user_data_train[u]['y'].append(data['user_data'][u]['y'][j])
                            elif test_blist[j]:
                                user_data_test[u]['x'].append(data['user_data'][u]['x'][j])
                                user_data_test[u]['y'].append(data['user_data'][u]['y'][j])

            users = [data['users'][i] for i in user_indices]
            all_data_train = {
                'users': users,
                'num_samples': num_samples_train,
                'user_data': user_data_train,
            }
            all_data_test = {
                'users': users,
                'num_samples': num_samples_test,
                'user_data': user_data_test,
            }
            if include_hierarchy:
                hierarchies = [data['hierarchies'][i] for i in user_indices]
                all_data_train['hierarchies'] = hierarchies
                all_data_test['hierarchies'] = hierarchies

            file_name_train = '%s_train_%s.json' % ((f[:-5]), arg_label)
            file_name_test = '%s_test_%s.json' % ((f[:-5]), arg_label)
            out_dir_train = os.path.join(train_dir, file_name_train)
            out_dir_test = os.path.join(test_dir, file_name_test)

            with open(out_dir_train, 'w') as fout:
                json.dump(all_data_train, fout)
            with open(out_dir_test, 'w') as fout:
                json.dump(all_data_test, fout)

        logging.info("Split.")

    # Equivalent to executing `./preprocess.sh -s niid --sf 1.0 -k 0 -t sample`
    # at `~/data/femnist/` folder of LEAF
    def preprocess_data(self, data_path):
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        # PART ONE: Downloading
        # correspond to ~/data/femnist/preprocess/get_data.sh in LEAF
        raw_dir = os.path.join(data_path, 'raw')
        class_dir = os.path.join(raw_dir, 'by_class')
        write_dir = os.path.join(raw_dir, 'by_write')

        if not os.path.isdir(class_dir) or not os.path.isdir(write_dir):
            logging.info("[FEMNIST Phase 1] Downloading the FEMNIST dataset. This may take a while.")
            if not os.path.isdir(class_dir):
                by_class_url = "https://s3.amazonaws.com/nist-srd/SD19/by_class.zip"
                self.download(by_class_url, raw_dir)
            if not os.path.isdir(write_dir):
                by_write_url = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"
                self.download(by_write_url, raw_dir)
            logging.info("[FEMNIST Phase 1] Downloaded.")

        # PART TWO: Downloading
        intermediate_dir = os.path.join(data_path, 'intermediate')
        if not os.path.exists(intermediate_dir):
            os.makedirs(intermediate_dir)
        all_data_dir = os.path.join(intermediate_dir, 'all_data')

        if not os.path.isdir(all_data_dir):
            logging.info("[FEMNIST Phase 2] Converting the raw data to files in json format. "
                         "This may take a while.")

            # correspond to ~/data/femnist/preprocess/get_file_dirs.py in LEAF
            class_file_dirs_path = os.path.join(intermediate_dir, 'class_file_dirs.pkl')
            if not os.path.isfile(class_file_dirs_path):
                self.extract_class_file_dirs(raw_dir, class_file_dirs_path)
            write_file_dirs_path = os.path.join(intermediate_dir, 'write_file_dirs.pkl')
            if not os.path.isfile(write_file_dirs_path):
                self.extract_write_file_dirs(raw_dir, write_file_dirs_path)

            # correspond to ~/data/femnist/preprocess/get_hashes.py
            class_file_hashes_path = os.path.join(intermediate_dir,
                                                  'class_file_hashes.pkl')
            if not os.path.isfile(class_file_hashes_path):
                self.compute_file_hashes(class_file_dirs_path, class_file_hashes_path,
                                         'by_class')
            write_file_hashes_path = os.path.join(intermediate_dir,
                                                  'write_file_hashes.pkl')
            if not os.path.isfile(write_file_hashes_path):
                self.compute_file_hashes(write_file_dirs_path, write_file_hashes_path,
                                         'by_write')

            # correspond to ~/data/femnist/preprocess/match_hashes.py
            write_with_class_path = os.path.join(intermediate_dir,
                                                 'write_with_class.pkl')
            if not os.path.isfile(write_with_class_path):
                self.match_hashes(class_file_hashes_path, write_file_hashes_path,
                                  write_with_class_path)

            # correspond to ~/data/femnist/preprocess/group_by_writer.py
            images_by_writer_path = os.path.join(intermediate_dir,
                                                 'images_by_writer.pkl')
            if not os.path.isfile(images_by_writer_path):
                self.group_by_writers(write_with_class_path, images_by_writer_path)

            # correspond to ~/data/femnist/preprocess/data_to_json.py
            self.summarize_all_data(images_by_writer_path, all_data_dir)

            logging.info("[FEMNIST Phase 2] Converted.")

        # PART THREE: Partitioning
        logging.info("[FEMNIST Phase 3] Partitioning the FEMNIST dataset. This may take a while.")

        # correspond to ~/data/utils/sample.py
        sampled_data_dir = os.path.join(intermediate_dir, 'sampled_data')
        if not os.path.isdir(sampled_data_dir):
            sample_rng_seed, scale = 233, 1.0  # TODO: to get rid of hard-coding
            self.sample_data(all_data_dir, sampled_data_dir, scale, sample_rng_seed)

        # correspond to ~/data/utils/split_data.py
        train_dir = os.path.join(data_path, 'ready', 'train')
        test_dir = os.path.join(data_path, 'ready', 'test')
        if os.path.isdir(train_dir):
            shutil.rmtree(train_dir)
        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir)
        split_rng_seed, train_fraction = 233, 0.9  # TODO: to get rid of hard-coding
        self.split_data(sampled_data_dir, train_dir, test_dir, train_fraction, split_rng_seed)

        logging.info("[FEMNIST Phase 3] Partitioned.")

    def num_train_examples(self):
        return self.trainset_size

    def num_test_examples(self):
        return self.testset_size
