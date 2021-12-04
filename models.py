import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config_global import DEVICE
from configs.config_model import RNNSTSPConfig


# Vanilla RNNCell
class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity="tanh", ct=False):
        super(VanillaRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.ct = ct

        self.weight_ih = nn.Parameter(torch.zeros((input_size, hidden_size)))
        self.weight_hh = nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()

        if self.nonlinearity == "tanh":
            self.act = F.tanh
        elif self.nonlinearity == "relu":
            self.act = F.relu
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inp, hidden_in):
        if not self.ct:
            hidden_out = self.act(torch.matmul(inp, self.weight_ih)
                                  + torch.matmul(hidden_in, self.weight_hh)
                                  + self.bias)
        else:
            alpha = 0.1
            hidden_out = (1 - alpha) * hidden_in \
                         + alpha * self.act(torch.matmul(inp, self.weight_ih)
                                            + torch.matmul(hidden_in, self.weight_hh)
                                            + self.bias)
        return hidden_out

    def init_hidden(self, batch_s):
        return torch.zeros(batch_s, self.hidden_size).to(DEVICE)


# RNN with short-term synaptic plasticity cell
# adapted from https://github.com/nmasse/Short-term-plasticity-RNN
class RNNSTSPCell(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity="relu"):
        super(RNNSTSPCell, self).__init__()
        self.syncfg = RNNSTSPConfig(input_size, hidden_size)

        self.input_size = self.syncfg.n_input
        self.hidden_size = self.syncfg.n_hidden
        self.nonlinearity = nonlinearity

        self.weight_ih = nn.Parameter(torch.from_numpy(self.syncfg.w_in0))
        self.weight_hh = nn.Parameter(torch.from_numpy(self.syncfg.w_rnn0))
        self.bias = nn.Parameter(torch.from_numpy(self.syncfg.b_rnn0))
        self.EI = self.syncfg.EI
        if self.EI:
            # ensure excitatory neurons only have positive outgoing weights,
            # and inhibitory neurons have negative outgoing weights
            self.EI_matrix = torch.from_numpy(self.syncfg.EI_matrix).to(DEVICE)
        self.w_rnn_mask = torch.from_numpy(self.syncfg.w_rnn_mask).to(DEVICE)

        self.alpha_stf = torch.from_numpy(self.syncfg.alpha_stf).to(DEVICE)
        self.alpha_std = torch.from_numpy(self.syncfg.alpha_std).to(DEVICE)
        self.U = torch.from_numpy(self.syncfg.U).to(DEVICE)
        self.dynamic_synapse = torch.from_numpy(self.syncfg.dynamic_synapse).to(DEVICE)

        if self.nonlinearity == "tanh":
            self.act = F.tanh
        elif self.nonlinearity == "relu":
            self.act = F.relu
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

    def forward(self, inp, hidden_in):
        h_in, syn_x, syn_u = hidden_in

        # Update the synaptic plasticity parameters
        # implement both synaptic short term facilitation and depression
        syn_x = syn_x + (self.alpha_std * (1.0 - syn_x)
                         - self.syncfg.dt_sec * syn_u * syn_x * h_in) * self.dynamic_synapse
        syn_u = syn_u + (self.alpha_stf * (self.U - syn_u)
                         + self.syncfg.dt_sec * self.U * (1.0 - syn_u) * h_in) * self.dynamic_synapse
        syn_x = torch.clamp(syn_x, min=0.0, max=1.0)
        syn_u = torch.clamp(syn_u, min=0.0, max=1.0)
        h_post = syn_u * syn_x * h_in

        # Update the hidden state. Only use excitatory projections from input layer to RNN
        # All input and RNN activity will be non-negative
        # no adding noise as original implementation
        if self.EI:
            eff_rnn_w = self.w_rnn_mask * torch.matmul(self.EI_matrix, F.relu(self.weight_hh))
        else:
            eff_rnn_w = self.w_rnn_mask * self.weight_hh
        h_out = h_in * (1 - self.syncfg.alpha_neuron) \
                + self.syncfg.alpha_neuron \
                * self.act(torch.matmul(inp, F.relu(self.weight_ih))
                           + torch.matmul(h_post, eff_rnn_w)
                           + self.bias)
        hidden_out = h_out, syn_x, syn_u
        return hidden_out

    def init_hidden(self, batch_s):
        # h_init is not optimized, different from original implementation
        h_out_init = torch.from_numpy(self.syncfg.h0).to(DEVICE).repeat(batch_s, 1)
        syn_x_init = torch.from_numpy(self.syncfg.syn_x_init).to(DEVICE).repeat(batch_s, 1)
        syn_u_init = torch.from_numpy(self.syncfg.syn_u_init).to(DEVICE).repeat(batch_s, 1)
        hidden_init = h_out_init, syn_x_init, syn_u_init
        return hidden_init


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_type):
        super(RNNModel, self).__init__()
        assert rnn_type in ['plainRNN', 'VanillaRNN', 'CTRNN', 'LSTM', 'GRU', 'RNNSTSP'],\
            'Given RNN type must be implemented'
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        if self.rnn_type == 'plainRNN':
            self.rnn = nn.RNNCell(input_size, hidden_size)
        elif self.rnn_type == 'VanillaRNN':
            self.rnn = VanillaRNNCell(input_size, hidden_size)
        elif self.rnn_type == 'CTRNN':
            self.rnn = VanillaRNNCell(input_size, hidden_size, ct=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTMCell(input_size, hidden_size)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRUCell(input_size, hidden_size)
        elif self.rnn_type == 'RNNSTSP':
            self.rnn = RNNSTSPCell(input_size, hidden_size)
        else:
            raise NotImplementedError('RNN cell not implemented')
        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inp, hidden_in):
        hid_out = self.rnn(inp, hidden_in)
        if self.rnn_type in ['LSTM', 'RNNSTSP']:
            rnn_out = hid_out[0]
        else:
            rnn_out = hid_out
        otp = self.out_layer(rnn_out)
        return otp, hid_out

    def init_hidden(self, batch_s):
        if self.rnn_type == 'LSTM':
            init_hid = (torch.zeros(batch_s, self.hidden_size).to(DEVICE),
                        torch.zeros(batch_s, self.hidden_size).to(DEVICE))
        elif self.rnn_type in ['plainRNN', 'GRU']:
            init_hid = torch.zeros(batch_s, self.hidden_size).to(DEVICE)
        elif self.rnn_type in ['VanillaRNN', 'CTRNN', 'RNNSTSP']:
            init_hid = self.rnn.init_hidden(batch_s)
        else:
            raise NotImplementedError('RNN init not implemented')
        return init_hid
