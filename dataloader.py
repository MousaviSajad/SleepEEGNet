'''
https://github.com/akaraspt/deepsleepnet
Copyright 2017 Akara Supratak and Hao Dong.  All rights reserved.
'''
import os
import numpy as np
import re
class SeqDataLoader(object):
    def __init__(self, data_dir, n_folds, fold_idx,classes):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.classes = classes

    def load_npz_file(self, npz_file):
        """Load data_2013 and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def save_to_npz_file(self, data, labels, sampling_rate, filename):

        # Save
        save_dict = {
            "x": data,
            "y": labels,
            "fs": sampling_rate,

        }
        np.savez(filename, **save_dict)
    def _load_npz_list_files(self, npz_files):
        """Load data_2013 and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print ("Loading {} ...".format(npz_f))
            tmp_data, tmp_labels, self.sampling_rate = self.load_npz_file(npz_f)
            if fs is None:
                fs = self.sampling_rate
            elif fs != self.sampling_rate:
                raise Exception("Found mismatch in sampling rate.")

            # Reshape the data_2013 to match the input of the model - conv2d
            tmp_data = np.squeeze(tmp_data)
            # tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

            # # Reshape the data_2013 to match the input of the model - conv1d
            # tmp_data = tmp_data[:, :, np.newaxis]

            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)

            # normalize each 30s sample such that each has zero mean and unit vairance
            tmp_data = (tmp_data - np.expand_dims(tmp_data.mean(axis=1),axis= 1)) / np.expand_dims(tmp_data.std(axis=1),axis=1)


            data.append(tmp_data)
            labels.append(tmp_labels)

        return data, labels

    def _load_cv_data(self, list_files):
        """Load sequence training and cross-validation sets."""
        # Split files for training and validation sets
        val_files = np.array_split(list_files, self.n_folds)
        train_files = np.setdiff1d(list_files, val_files[self.fold_idx])

        # Load a npz file
        print ("Load training set:")
        data_train, label_train = self._load_npz_list_files(train_files)
        print (" ")
        print ("Load validation set:")
        data_val, label_val = self._load_npz_list_files(val_files[self.fold_idx])
        print (" ")

        return data_train, label_train, data_val, label_val

    def load_test_data(self):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        # Files for validation sets
        val_files = np.array_split(npzfiles, self.n_folds)
        val_files = val_files[self.fold_idx]

        print ("\n========== [Fold-{}] ==========\n".format(self.fold_idx))

        print ("Load validation set:")
        data_val, label_val = self._load_npz_list_files(val_files)

        return data_val, label_val

    def load_data(self, seq_len = 10, shuffle = True, n_files=None):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        if n_files is not None:
            npzfiles = npzfiles[:n_files]

        # subject_files = []
        # for idx, f in enumerate(allfiles):
        #     if self.fold_idx < 10:
        #         pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
        #     else:
        #         pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
        #     if pattern.match(f):
        #         subject_files.append(os.path.join(self.data_dir, f))

        # randomize the order of the file names just for one time!
        r_permute = np.random.permutation(len(npzfiles))
        filename = "r_permute.npz"
        if (os.path.isfile(filename)):
            with np.load(filename) as f:
                r_permute = f["inds"]
        else:
            save_dict = {
                "inds": r_permute,

            }
            np.savez(filename, **save_dict)

        npzfiles = np.asarray(npzfiles)[r_permute]
        train_files = np.array_split(npzfiles, self.n_folds)
        subject_files = train_files[self.fold_idx]


        train_files = list(set(npzfiles) - set(subject_files))
        # train_files.sort()
        # subject_files.sort()

        # Load training and validation sets
        print ("\n========== [Fold-{}] ==========\n".format(self.fold_idx))
        print ("Load training set:")
        data_train, label_train = self._load_npz_list_files(train_files)
        print (" ")
        print ("Load Test set:")
        data_test, label_test = self._load_npz_list_files(subject_files)
        print (" ")

        print ("Training set: n_subjects={}".format(len(data_train)))
        n_train_examples = 0
        for d in data_train:
            print d.shape
            n_train_examples += d.shape[0]
        print ("Number of examples = {}".format(n_train_examples))
        self.print_n_samples_each_class(np.hstack(label_train),self.classes)
        print (" ")
        print ("Test set: n_subjects = {}".format(len(data_test)))
        n_test_examples = 0
        for d in data_test:
            print d.shape
            n_test_examples += d.shape[0]
        print ("Number of examples = {}".format(n_test_examples))
        self.print_n_samples_each_class(np.hstack(label_test),self.classes)
        print (" ")

        data_train = np.vstack(data_train)
        label_train = np.hstack(label_train)
        data_train = [data_train[i:i + seq_len] for i in range(0, len(data_train), seq_len)]
        label_train = [label_train[i:i + seq_len] for i in range(0, len(label_train), seq_len)]
        if data_train[-1].shape[0]!=seq_len:
            data_train.pop()
            label_train.pop()

        data_train = np.asarray(data_train)
        label_train = np.asarray(label_train)

        data_test = np.vstack(data_test)
        label_test = np.hstack(label_test)
        data_test = [data_test[i:i + seq_len] for i in range(0, len(data_test), seq_len)]
        label_test = [label_test[i:i + seq_len] for i in range(0, len(label_test), seq_len)]

        if data_test[-1].shape[0]!=seq_len:
            data_test.pop()
            label_test.pop()

        data_test = np.asarray(data_test)
        label_test = np.asarray(label_test)

        # shuffle
        if shuffle is True:
            # training data_2013
            permute = np.random.permutation(len(label_train))
            data_train = np.asarray(data_train)
            data_train = data_train[permute]
            label_train = label_train[permute]

            # test data_2013
            permute = np.random.permutation(len(label_test))
            data_test = np.asarray(data_test)
            data_test = data_test[permute]
            label_test = label_test[permute]

        return data_train, label_train, data_test, label_test

    @staticmethod
    def load_subject_data(data_dir, subject_idx):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(data_dir)
        subject_files = []
        for idx, f in enumerate(allfiles):
            if subject_idx < 10:
                pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(subject_idx))
            else:
                pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(subject_idx))
            if pattern.match(f):
                subject_files.append(os.path.join(data_dir, f))

        # Files for validation sets
        if len(subject_files) == 0 or len(subject_files) > 2:
            raise Exception("Invalid file pattern")

        def load_npz_file(npz_file):
            """Load data_2013 and labels from a npz file."""
            with np.load(npz_file) as f:
                data = f["x"]
                labels = f["y"]
                sampling_rate = f["fs"]
            return data, labels, sampling_rate

        def load_npz_list_files(npz_files):
            """Load data_2013 and labels from list of npz files."""
            data = []
            labels = []
            fs = None
            for npz_f in npz_files:
                print ("Loading {} ...".format(npz_f))
                tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
                if fs is None:
                    fs = sampling_rate
                elif fs != sampling_rate:
                    raise Exception("Found mismatch in sampling rate.")

                # Reshape the data_2013 to match the input of the model - conv2d
                tmp_data = np.squeeze(tmp_data)
                # tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]

                # # Reshape the data_2013 to match the input of the model - conv1d
                # tmp_data = tmp_data[:, :, np.newaxis]

                # Casting
                tmp_data = tmp_data.astype(np.float32)
                tmp_labels = tmp_labels.astype(np.int32)

                data.append(tmp_data)
                labels.append(tmp_labels)

            return data, labels

        print ("Load data_2013 from: {}".format(subject_files))
        data, labels = load_npz_list_files(subject_files)

        return data, labels

    @staticmethod
    def print_n_samples_each_class(labels,classes):
        class_dict = dict(zip(range(len(classes)),classes))
        unique_labels = np.unique(labels)
        for c in unique_labels:
            n_samples = len(np.where(labels == c)[0])
            print ("{}: {}".format(class_dict[c], n_samples))