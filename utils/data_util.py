import os

import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence

from configs.config_global import ROOT_DIR

char_list = ['a', 'b', 'c', 'd', 'e',
             'g', 'h', 'l', 'm', 'n',
             'o', 'p', 'q', 'r', 's',
             'u', 'v', 'w', 'y', 'z']


def num2chr(num):
    return char_list[num]


# get subset of the original dataset based on target
# example usage: train_data = get_subset(train_data, include_targets)
def get_subset(data_set, inc_targets):
    bool_idx = torch.tensor([(tar in inc_targets) for tar in data_set.targets])
    subset_idx = torch.nonzero(bool_idx, as_tuple=True)[0]
    data_set = Subset(data_set, subset_idx)
    return data_set


class SkipTransform(object):
    def __init__(self, skip_num):
        assert type(skip_num) is int and skip_num >= 1
        self.skip_num = skip_num

    def __call__(self, tra):
        return tra[:, ::self.skip_num]


# trim leading and trailing zeros from a NumPy array in the second dimension
class TrimZeros(object):
    def __call__(self, tra):
        nz = np.nonzero(tra)
        tra_trimmed = tra[:, nz[1].min():nz[1].max() + 1]
        return tra_trimmed


# perform accumulated sum to turn velocity position, along second dimension
class CumSum(object):
    def __call__(self, tra):
        return np.cumsum(tra, axis=1)


def pos_interp(trajectory, t_scale):
    ret_trajectory = []
    for row in range(trajectory.shape[0]):
        row_data = trajectory[row]
        len_data = len(row_data)
        ret_row = np.interp(np.linspace(0, len_data - 1, round(len_data / t_scale)),
                            np.linspace(0, len_data - 1, len_data), row_data)
        ret_trajectory.append(ret_row)
    return np.stack(ret_trajectory)


# stretch position data, linear interpolation
class TimeScalePos(object):
    def __init__(self, t_scale):
        self.t_scale = t_scale

    def __call__(self, tra):
        if self.t_scale == 1.0:
            ret_tra = tra
        else:
            ret_tra = pos_interp(tra, t_scale=self.t_scale)
        return ret_tra


class TimeScaleVel(object):
    def __init__(self, t_scale):
        self.t_scale = t_scale

    def __call__(self, tra):
        if self.t_scale == 1.0:
            ret_tra = tra
        else:
            pos_tra = np.cumsum(tra, axis=1)
            int_pos_tra = pos_interp(pos_tra, t_scale=self.t_scale)
            ret_tra = np.diff(int_pos_tra, axis=1, prepend=0)
        return ret_tra


# scale
class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, tra):
        return tra * self.scale


def pad_collate(batch):
    # A data tuple has the form: (trajectory, label)
    tensors, labels = [], []
    # Gather in lists, and encode labels as indices
    for tra, label in batch:
        tensors += [torch.tensor(tra.transpose(), dtype=torch.float)]
        labels += [torch.tensor(label, dtype=torch.long)]

    # Group the list of tensors into a batched tensor
    tensors = [tensor.flip(dims=(0, )) for tensor in tensors]
    tensors = pad_sequence(tensors)
    tensors = tensors.flip(dims=(0, ))
    labels = torch.stack(labels)
    return tensors, labels


class CharacterTrajectoriesDataset(Dataset):
    def __init__(self, large_split, transform=None, target_transform=None):
        mat_fname = os.path.join(ROOT_DIR, 'data', 'mtsdata',
                                 'CharacterTrajectories', 'CharacterTrajectories.mat')
        mat_contents = sio.loadmat(mat_fname, simplify_cells=True)
        self.transform = transform
        self.target_transform = target_transform

        # use small split of the dataset, 300 examples
        if not large_split:
            load_str = 'train'
        # use large split of the dataset, 2558 example
        else:
            load_str = 'test'
        self.data = mat_contents['mts'][load_str]
        # change the original label to the range [0, 19]
        self.label = mat_contents['mts'][load_str + 'labels'] - 1
        self.targets = self.label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        trajectory = self.data[idx]
        label = self.label[idx]
        if self.transform:
            trajectory = self.transform(trajectory)
        if self.target_transform:
            label = self.target_transform(label)
        return trajectory, label
