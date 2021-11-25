import os
import scipy.io as sio
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

# device to run algorithm on
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, 20)

    def forward(self, inp, hidden_in):
        hid_out = self.rnn(inp, hidden_in)
        otp = self.out_layer(hid_out[0])
        return otp, hid_out

    def init_hidden(self, batch_s):
        init_hid = (torch.zeros(batch_s, self.hidden_size).to(DEVICE),
                    torch.zeros(batch_s, self.hidden_size).to(DEVICE))
        return init_hid


def pad_collate(batch):
    # A data tuple has the form:
    # trajectory, label
    tensors, labels = [], []
    # Gather in lists, and encode labels as indices
    for tra, label in batch:
        tensors += [torch.tensor(tra.transpose(), dtype=torch.float)]
        labels += [torch.tensor(label-1, dtype=torch.int64)]

    # Group the list of tensors into a batched tensor
    # TODO: pad sequence at the begining
    tensors = pad_sequence(tensors)
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
        self.label = mat_contents['mts'][load_str + 'labels']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        trajectory = self.data[idx]
        label = self.label[idx]
        return trajectory, label


if __name__ == '__main__':
    dataset = CharacterTrajectoriesDataset()

    batch_s = 20
    dataloader = DataLoader(dataset, batch_size=batch_s, shuffle=False, collate_fn=pad_collate)
    criterion = nn.CrossEntropyLoss()
    model = RNNModel(3, 64)
    optimizer = torch.optim.Adam(model.parameters())
    for ep in range(100):
        for data, label in dataloader:
            loss = 0

            hid = model.init_hidden(batch_s)
            for t_ in range(data.shape[1]):
                output, hid = model(data[t_], hid)

                if t_ > data.shape[1] - 10:
                    loss += criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())


