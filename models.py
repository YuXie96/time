import torch
import torch.nn as nn
from configs.config_global import DEVICE


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_type='LSTM'):
        super(RNNModel, self).__init__()
        assert rnn_type in ['LSTM', 'GRU'], 'Given RNN type must be implemented'
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTMCell(input_size, hidden_size)
        else:
            self.rnn = nn.GRUCell(input_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inp, hidden_in):
        hid_out = self.rnn(inp, hidden_in)
        if self.rnn_type == 'LSTM':
            rnn_out = hid_out[0]
        else:
            rnn_out = hid_out
        otp = self.out_layer(rnn_out)
        return otp, hid_out

    def init_hidden(self, batch_s):
        if self.rnn_type == 'LSTM':
            init_hid = (torch.zeros(batch_s, self.hidden_size).to(DEVICE),
                        torch.zeros(batch_s, self.hidden_size).to(DEVICE))
        else:
            init_hid = torch.zeros(batch_s, self.hidden_size).to(DEVICE)
        return init_hid
