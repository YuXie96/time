import numpy as np


def initialize(dims, connection_prob=1.0, shape=0.1, scale=1.0):
    w = np.random.gamma(shape, scale, size=dims)
    w *= (np.random.rand(*dims) < connection_prob)

    return np.float32(w)


# configs for RNN with short-term synaptic plasticity cell
# adapted from https://github.com/nmasse/Short-term-plasticity-RNN
class RNNSTSPConfig:
    def __init__(self, in_size, h_size):

        self.n_input = in_size
        self.n_hidden = h_size  # default 100

        self.exc_inh_prop = 0.8
        self.balance_EI = True
        self.synapse_config = 'exc_dep_inh_fac'

        self.membrane_time_constant = 100
        self.tau_fast = 200
        self.tau_slow = 1500
        self.dt = 10  # default 10
        # The time step in seconds
        self.dt_sec = self.dt / 1000

        self.trial_length = 2500  # fix 500 + sample 500 + delay 1000 +  test 500
        # Length of each trial in time steps
        self.num_time_steps = self.trial_length // self.dt

        # If exc_inh_prop is < 1, then neurons can be either excitatory or
        # inhibitory; if exc_inh_prop = 1, then the weights projecting from
        # a single neuron can be a mixture of excitatory or inhibitory
        if self.exc_inh_prop < 1:
            self.EI = True
        else:
            self.EI = False

        self.num_exc_units = int(np.round(self.n_hidden * self.exc_inh_prop))
        self.num_inh_units = self.n_hidden - self.num_exc_units

        self.EI_list = np.ones(self.n_hidden, dtype=np.float32)
        self.EI_list[-self.num_inh_units:] = -1.0

        self.ind_inh = np.where(self.EI_list == -1)[0]

        self.EI_matrix = np.diag(self.EI_list)

        # Membrane time constant of RNN neurons
        self.alpha_neuron = np.float32(self.dt) / self.membrane_time_constant

        # initial neural activity
        self.h0 = 0.1 * np.ones((1, self.n_hidden), dtype=np.float32)

        # initial input weights
        self.w_in0 = initialize([self.n_input, self.n_hidden], shape=0.2, scale=1.0)

        # Initialize starting recurrent weights
        # If excitatory/inhibitory neurons desired,
        # initializes with random matrix with zeroes on the diagonal
        # If not, initializes with a diagonal matrix
        if self.EI:
            self.w_rnn0 = initialize([self.n_hidden, self.n_hidden])
            if self.balance_EI:
                # increase the weights to and from inh units to balance excitation and inhibition
                self.w_rnn0[:, self.ind_inh] = initialize([self.n_hidden, self.num_inh_units],
                                                          shape=0.2, scale=1.0)
                self.w_rnn0[self.ind_inh, :] = initialize([self.num_inh_units, self.n_hidden],
                                                          shape=0.2, scale=1.0)

        else:
            self.w_rnn0 = 0.54 * np.eye(self.n_hidden)

        # initial recurrent biases
        self.b_rnn0 = np.zeros(self.n_hidden, dtype=np.float32)

        # for EI networks, masks will prevent self-connections, and inh to output connections
        self.w_rnn_mask = np.ones_like(self.w_rnn0)
        if self.EI:
            self.w_rnn_mask = np.ones((self.n_hidden, self.n_hidden), dtype=np.float32)\
                              - np.eye(self.n_hidden, dtype=np.float32)

        self.w_rnn0 *= self.w_rnn_mask

        synaptic_configurations = {
            'full': ['facilitating' if i % 2 == 0 else 'depressing' for i in range(self.n_hidden)],
            'fac': ['facilitating' for i in range(self.n_hidden)],
            'dep': ['depressing' for i in range(self.n_hidden)],
            'exc_fac': ['facilitating' if self.EI_list[i] == 1 else 'static' for i in range(self.n_hidden)],
            'exc_dep': ['depressing' if self.EI_list[i] == 1 else 'static' for i in range(self.n_hidden)],
            'inh_fac': ['facilitating' if self.EI_list[i] == -1 else 'static' for i in range(self.n_hidden)],
            'inh_dep': ['depressing' if self.EI_list[i] == -1 else 'static' for i in range(self.n_hidden)],
            'exc_dep_inh_fac': ['depressing' if self.EI_list[i] == 1 else 'facilitating' for i in
                                range(self.n_hidden)]
        }

        # initialize synaptic values
        self.alpha_stf = np.ones((1, self.n_hidden), dtype=np.float32)
        self.alpha_std = np.ones((1, self.n_hidden), dtype=np.float32)
        self.U = np.ones((1, self.n_hidden), dtype=np.float32)
        self.syn_x_init = np.ones((1, self.n_hidden), dtype=np.float32)
        self.syn_u_init = 0.3 * np.ones((1, self.n_hidden), dtype=np.float32)
        self.dynamic_synapse = np.ones((1, self.n_hidden), dtype=np.float32)

        for i in range(self.n_hidden):
            if self.synapse_config not in synaptic_configurations.keys():
                self.dynamic_synapse[0, i] = 0
            elif synaptic_configurations[self.synapse_config][i] == 'facilitating':
                self.alpha_stf[0, i] = self.dt / self.tau_slow
                self.alpha_std[0, i] = self.dt / self.tau_fast
                self.U[0, i] = 0.15
                self.syn_u_init[0, i] = self.U[0, i]
                self.dynamic_synapse[0, i] = 1

            elif synaptic_configurations[self.synapse_config][i] == 'depressing':
                self.alpha_stf[0, i] = self.dt / self.tau_fast
                self.alpha_std[0, i] = self.dt / self.tau_slow
                self.U[0, i] = 0.45
                self.syn_u_init[0, i] = self.U[0, i]
                self.dynamic_synapse[0, i] = 1
