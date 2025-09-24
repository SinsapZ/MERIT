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
            for trial_feature in subject_feature:
                if j + 1 in ids:
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


class ADFDLoader(APAVALoader):
    pass


class ADFDDependentLoader(APAVALoader):
    pass


class PTBLoader(APAVALoader):
    pass


class PTBXLLoader(APAVALoader):
    pass


