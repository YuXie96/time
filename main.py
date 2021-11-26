import os
import scipy.io as sio
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

# device to run algorithm on
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


def grad_clipping(model, max_norm, printing=False):
    p_req_grad = [p for p in model.parameters() if p.requires_grad]

    if printing:
        grad_before = 0.0
        for p in p_req_grad:
            param_norm = p.grad.data.norm(2)
            grad_before += param_norm.item() ** 2
        grad_before = grad_before ** (1. / 2)

    clip_grad_norm_(p_req_grad, max_norm)

    if printing:
        grad_after = 0.0
        for p in p_req_grad:
            param_norm = p.grad.data.norm(2)
            grad_after += param_norm.item() ** 2
        grad_after = grad_after ** (1. / 2)

        if grad_before > grad_after:
            print("clipped")
            print("before: ", grad_before)
            print("after: ", grad_after)


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_LSTM=True):
        super(RNNModel, self).__init__()
        self.use_LSTM = use_LSTM
        self.hidden_size = hidden_size
        if self.use_LSTM:
            self.rnn = nn.LSTMCell(input_size, hidden_size)
        else:
            self.rnn = nn.GRUCell(input_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inp, hidden_in):
        hid_out = self.rnn(inp, hidden_in)
        if self.use_LSTM:
            rnn_out = hid_out[0]
        else:
            rnn_out = hid_out
        otp = self.out_layer(rnn_out)
        return otp, hid_out

    def init_hidden(self, batch_s):
        if self.use_LSTM:
            init_hid = (torch.zeros(batch_s, self.hidden_size).to(DEVICE),
                        torch.zeros(batch_s, self.hidden_size).to(DEVICE))
        else:
            init_hid = torch.zeros(batch_s, self.hidden_size).to(DEVICE)
        return init_hid


def pad_collate(batch):
    # A data tuple has the form:
    # trajectory, label
    tensors, labels = [], []
    # Gather in lists, and encode labels as indices
    for tra, label in batch:
        tensors += [torch.tensor(tra.transpose(), dtype=torch.float)]
        labels += [torch.tensor(label, dtype=torch.int64)]

    # Group the list of tensors into a batched tensor
    # TODO: normalize tensors
    # tensors = [tensor.flip(dims=(0, )) for tensor in tensors]
    tensors = pad_sequence(tensors)
    # tensors = tensors.flip(dims=(0, ))
    labels = torch.stack(labels)
    return tensors, labels


class CharacterTrajectoriesDataset(Dataset):
    def __init__(self, train=True):
        mat_fname = os.path.join(ROOT_DIR, 'data', 'mtsdata', 'CharacterTrajectories', 'CharacterTrajectories.mat')
        mat_contents = sio.loadmat(mat_fname, simplify_cells=True)
        if train:
            load_str = 'train'
        else:
            load_str = 'test'
        self.data = mat_contents['mts'][load_str]
        self.label = mat_contents['mts'][load_str + 'labels'] - 1
        self.targets = self.label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        trajectory = self.data[idx]
        label = self.label[idx]
        return trajectory, label


def get_subset(data_set, inc_targets):
    bool_idx = torch.tensor([(tar in inc_targets) for tar in data_set.targets])
    subset_idx = torch.nonzero(bool_idx, as_tuple=True)[0]
    data_set = Subset(data_set, subset_idx)
    return data_set


if __name__ == '__main__':
    readout_steps = 10
    num_classes = 5
    include_targets = list(range(num_classes))
    #
    # include_targets = [1, 4, 7, 14, 16, 18, 19]
    # num_classes = len(include_targets)

    train_data = CharacterTrajectoriesDataset(train=False)
    test_data = CharacterTrajectoriesDataset()

    train_data = get_subset(train_data, include_targets)
    test_data = get_subset(test_data, include_targets)

    batch_s = 20
    train_loader = DataLoader(train_data, batch_size=batch_s, collate_fn=pad_collate, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_s, shuffle=False, collate_fn=pad_collate, drop_last=True)

    criterion = nn.CrossEntropyLoss()
    model = RNNModel(3, 64, num_classes)
    optimizer = torch.optim.Adam(model.parameters())

    for ep in range(100):
        for data, label in train_loader:
            loss = 0.0
            hid = model.init_hidden(batch_s)
            for t_ in range(data.shape[1]):
                output, hid = model(data[t_], hid)

                if t_ >= data.shape[1] - readout_steps:
                    loss += criterion(output, label)
            optimizer.zero_grad()
            loss /= readout_steps
            grad_clipping(model, 1.0)
            loss.backward()
            optimizer.step()

        print(loss.item())
        correct = 0
        total = 0
        with torch.no_grad():
            for data, label in test_loader:
                hid = model.init_hidden(batch_s)
                for t_ in range(data.shape[1]):
                    output, hid = model(data[t_], hid)
                    if t_ >= data.shape[1] - readout_steps:
                        _, predicted = torch.max(output.data, 1)
                        total += label.size(0)
                        correct += (predicted == label).sum().item()

        print('Accuracy of the network on the test set: %d %%' % (
                100 * correct / total))

    # prepare to count predictions for each class
    classes = list(range(num_classes))
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data, label in test_loader:
            hid = model.init_hidden(batch_s)
            for t_ in range(data.shape[1]):
                output, hid = model(data[t_], hid)
                if t_ >= data.shape[1] - readout_steps:
                    _, predictions = torch.max(output.data, 1)
                    # collect the correct predictions for each class
                    for lab, prediction in zip(label, predictions):
                        if lab == prediction:
                            correct_pred[classes[lab]] += 1
                        total_pred[classes[lab]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {} is: {:.1f} %".format(classname, accuracy))
