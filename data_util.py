import os

import scipy.io as sio
import torch
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence

from configs.config_global import ROOT_DIR


def get_subset(data_set, inc_targets):
    bool_idx = torch.tensor([(tar in inc_targets) for tar in data_set.targets])
    subset_idx = torch.nonzero(bool_idx, as_tuple=True)[0]
    data_set = Subset(data_set, subset_idx)
    return data_set


def pad_collate(batch):
    # A data tuple has the form:
    # trajectory, label
    tensors, labels = [], []
    # Gather in lists, and encode labels as indices
    for tra, label in batch:
        tra = tra[:, ::8]
        tensors += [torch.tensor(tra.transpose(), dtype=torch.float)]
        labels += [torch.tensor(label, dtype=torch.int64)]

    # Group the list of tensors into a batched tensor
    # TODO: normalize tensors based on position, add fixation and readout cue
    tensors = [tensor.flip(dims=(0, )) for tensor in tensors]
    tensors = pad_sequence(tensors)
    tensors = tensors.flip(dims=(0, ))
    labels = torch.stack(labels)
    return tensors, labels


class CharacterTrajectoriesDataset(Dataset):
    def __init__(self, train=True):
        mat_fname = os.path.join(ROOT_DIR, 'data', 'mtsdata',
                                 'CharacterTrajectories', 'CharacterTrajectories.mat')
        mat_contents = sio.loadmat(mat_fname, simplify_cells=True)
        if train:
            load_str = 'train'
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
        return trajectory, label
