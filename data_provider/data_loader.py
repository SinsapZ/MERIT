import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.utils import shuffle
from .uea import normalize_batch_ts


class APAVALoader(Dataset):
    def __init__(self, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")
        data_list = np.load(self.label_path)
        all_ids = list(data_list[:, 1])
        val_ids = [15, 16, 19, 20]
        test_ids = [1, 2, 17, 18]
        train_ids = [int(i) for i in all_ids if i not in val_ids + test_ids]
        self.train_ids, self.val_ids, self.test_ids = train_ids, val_ids, test_ids
        self.X, self.y = self.load_apava(self.data_path, self.label_path, flag=flag)
        # remap labels to contiguous [0, K-1]
        uniq = np.unique(self.y)
        remap = {v: i for i, v in enumerate(sorted(uniq))}
        self.y = np.vectorize(remap.get)(self.y).astype(np.int64)
        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]

    def load_apava(self, data_path, label_path, flag=None):
        feature_list = []
        label_list = []
        filenames = []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames.sort()
        if flag == "TRAIN":
            ids = self.train_ids
            print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids
            print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids
            print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            sid = int(trial_label[1])
            for trial_feature in subject_feature:
                if sid in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)
        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


# The following dataset classes are simplified placeholders. Extend as needed.
class TDBRAINLoader(APAVALoader):
    pass


class ADFDLoader(Dataset):
    def __init__(self, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")
        a, b = 0.6, 0.8
        self.train_ids, self.val_ids, self.test_ids = self._split_ids(self.label_path, a, b)
        self.X, self.y = self._load_by_ids(self.data_path, self.label_path, flag=flag)
        uniq = np.unique(self.y)
        remap = {v: i for i, v in enumerate(sorted(uniq))}
        self.y = np.vectorize(remap.get)(self.y).astype(np.int64)
        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]

    def _split_ids(self, label_path, a=0.6, b=0.8):
        data_list = np.load(label_path)
        cn_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])
        ftd_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])
        ad_list = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])
        train_ids = (
            cn_list[: int(a * len(cn_list))]
            + ftd_list[: int(a * len(ftd_list))]
            + ad_list[: int(a * len(ad_list))]
        )
        val_ids = (
            cn_list[int(a * len(cn_list)) : int(b * len(cn_list))]
            + ftd_list[int(a * len(ftd_list)) : int(b * len(ftd_list))]
            + ad_list[int(a * len(ad_list)) : int(b * len(ad_list))]
        )
        test_ids = (
            cn_list[int(b * len(cn_list)) :]
            + ftd_list[int(b * len(ftd_list)) :]
            + ad_list[int(b * len(ad_list)) :]
        )
        return train_ids, val_ids, test_ids

    def _load_by_ids(self, data_path, label_path, flag=None):
        feature_list, label_list, filenames = [], [], []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames.sort()
        if flag == "TRAIN":
            ids = self.train_ids; print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids; print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids; print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            sid = int(trial_label[1])
            for trial_feature in subject_feature:
                if sid in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)
        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class ADFDDependentLoader(Dataset):
    def __init__(self, root_path, flag=None):
        from sklearn.model_selection import train_test_split
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/label.npy')
        # Load all samples (sample-dependent split)
        X_all, y_all = self._load_all(self.data_path, self.label_path)
        
        # Use stratified split like MedGNN to ensure balanced class distribution
        # Split: 60% train, 20% val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 * 0.8 = 0.2
        )
        
        if flag == 'TRAIN':
            self.X, self.y = X_train, y_train
            print(f'train samples: {len(y_train)}, class distribution: {np.bincount(y_train.astype(int))}')
        elif flag == 'VAL':
            self.X, self.y = X_val, y_val
            print(f'val samples: {len(y_val)}, class distribution: {np.bincount(y_val.astype(int))}')
        elif flag == 'TEST':
            self.X, self.y = X_test, y_test
            print(f'test samples: {len(y_test)}, class distribution: {np.bincount(y_test.astype(int))}')
        else:
            self.X, self.y = X_all, y_all
            
        # remap labels to contiguous [0, K-1]
        uniq = np.unique(self.y)
        remap = {v: i for i, v in enumerate(sorted(uniq))}
        self.y = np.vectorize(remap.get)(self.y).astype(np.int64)
        # normalize
        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]

    def _load_all(self, data_path, label_path):
        feature_list, label_list, filenames = [], [], []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames.sort()
        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)  # (num_trials, T, C)
            # append all trials with the subject label
            for trial_feature in subject_feature:
                feature_list.append(trial_feature)
                label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)[:, 0]
        return X, y

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class PTBLoader(Dataset):
    def __init__(self, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")
        a, b = 0.55, 0.7
        self.train_ids, self.val_ids, self.test_ids = self._split_ids(self.label_path, a, b)
        self.X, self.y = self._load_by_ids(self.data_path, self.label_path, flag=flag)
        uniq = np.unique(self.y)
        remap = {v: i for i, v in enumerate(sorted(uniq))}
        self.y = np.vectorize(remap.get)(self.y).astype(np.int64)
        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]

    def _split_ids(self, label_path, a=0.55, b=0.7):
        data_list = np.load(label_path)
        hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])
        my_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])
        train_ids = hc_list[: int(a * len(hc_list))] + my_list[: int(a * len(my_list))]
        val_ids = (
            hc_list[int(a * len(hc_list)) : int(b * len(hc_list))]
            + my_list[int(a * len(my_list)) : int(b * len(my_list))]
        )
        test_ids = hc_list[int(b * len(hc_list)) :] + my_list[int(b * len(my_list)) :]
        return train_ids, val_ids, test_ids

    def _load_by_ids(self, data_path, label_path, flag=None):
        feature_list, label_list, filenames = [], [], []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames.sort()
        if flag == "TRAIN":
            ids = self.train_ids; print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids; print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids; print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            sid = int(trial_label[1])
            for trial_feature in subject_feature:
                if sid in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)
        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


class PTBXLLoader(Dataset):
    def __init__(self, root_path, flag=None):
        self.root_path = root_path
        self.data_path = os.path.join(root_path, "Feature/")
        self.label_path = os.path.join(root_path, "Label/label.npy")
        a, b = 0.6, 0.8
        self.train_ids, self.val_ids, self.test_ids = self._split_ids(self.label_path, a, b)
        self.X, self.y = self._load_by_ids(self.data_path, self.label_path, flag=flag)
        uniq = np.unique(self.y)
        remap = {v: i for i, v in enumerate(sorted(uniq))}
        self.y = np.vectorize(remap.get)(self.y).astype(np.int64)
        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]

    def _split_ids(self, label_path, a=0.6, b=0.8):
        data_list = np.load(label_path)
        no_list  = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])
        mi_list  = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])
        sttc_list= list(data_list[np.where(data_list[:, 0] == 2)][:, 1])
        cd_list  = list(data_list[np.where(data_list[:, 0] == 3)][:, 1])
        hyp_list = list(data_list[np.where(data_list[:, 0] == 4)][:, 1])
        train_ids = (
            no_list[: int(a * len(no_list))]
            + mi_list[: int(a * len(mi_list))]
            + sttc_list[: int(a * len(sttc_list))]
            + cd_list[: int(a * len(cd_list))]
            + hyp_list[: int(a * len(hyp_list))]
        )
        val_ids = (
            no_list[int(a * len(no_list)) : int(b * len(no_list))]
            + mi_list[int(a * len(mi_list)) : int(b * len(mi_list))]
            + sttc_list[int(a * len(sttc_list)) : int(b * len(sttc_list))]
            + cd_list[int(a * len(cd_list)) : int(b * len(cd_list))]
            + hyp_list[int(a * len(hyp_list)) : int(b * len(hyp_list))]
        )
        test_ids = (
            no_list[int(b * len(no_list)) :]
            + mi_list[int(b * len(mi_list)) :]
            + sttc_list[int(b * len(sttc_list)) :]
            + cd_list[int(b * len(cd_list)) :]
            + hyp_list[int(b * len(hyp_list)) :]
        )
        return train_ids, val_ids, test_ids

    def _load_by_ids(self, data_path, label_path, flag=None):
        feature_list, label_list, filenames = [], [], []
        subject_label = np.load(label_path)
        for filename in os.listdir(data_path):
            filenames.append(filename)
        filenames.sort()
        if flag == "TRAIN":
            ids = self.train_ids; print("train ids:", ids)
        elif flag == "VAL":
            ids = self.val_ids; print("val ids:", ids)
        elif flag == "TEST":
            ids = self.test_ids; print("test ids:", ids)
        else:
            ids = subject_label[:, 1]
        for j in range(len(filenames)):
            trial_label = subject_label[j]
            path = data_path + filenames[j]
            subject_feature = np.load(path)
            sid = int(trial_label[1])
            for trial_feature in subject_feature:
                if sid in ids:
                    feature_list.append(trial_feature)
                    label_list.append(trial_label)
        X = np.array(feature_list)
        y = np.array(label_list)
        X, y = shuffle(X, y, random_state=42)
        return X, y[:, 0]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)


