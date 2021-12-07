"""Experiments and corresponding analysis.
format adapted from https://github.com/gyyang/olfaction_evolution

Each experiment is described by a function that returns a list of configurations
function name is the experiment name

combinatorial mode:
    config_ranges should not have repetitive values
sequential mode:
    config_ranges values should have equal length,
    otherwise this will only loop through the shortest one
control mode:
    base_config must contain keys in config_ranges
"""
import os
import copy
from collections import OrderedDict
import logging

import numpy as np

from configs.config_global import ROOT_DIR, LOG_LEVEL
from configs.configs import BaseConfig
from utils.config_utils import vary_config

from analysis.train_analysis import plot_train_log
import evaluate
from analysis import plots


def init_analysis(configs_):
    logging.basicConfig(level=LOG_LEVEL)
    exp_name = configs_[0].experiment_name
    print('Analyzing ' + exp_name)
    exp_path = os.path.join(ROOT_DIR, 'experiments', exp_name) + os.sep
    plot_train_log([exp_path], exp_name=exp_name)


# -----------------------------------------------------
# experiments
# -----------------------------------------------------


def timescale():
    config = BaseConfig()
    config.experiment_name = 'timescale'
    config.rnn_type = 'plainRNN'
    config.t_scale = 1.0
    config.augment = None
    config.use_velocity = False
    config.context = None
    config.context_w = 10
    config.hidden_size = 64

    config.num_ep = 40

    config_ranges = OrderedDict()
    config_ranges['rnn_type'] = ['plainRNN',
                                 'CTRNN',
                                 'LSTM',
                                 'GRU',
                                 'RNNSTSP']
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def timescale_aug():
    config = BaseConfig()
    config.experiment_name = 'timescale_aug'
    config.rnn_type = 'plainRNN'
    config.t_scale = 1.0
    config.augment = (0.5, 1.5)
    config.use_velocity = False
    config.context = None
    config.context_w = 10
    config.hidden_size = 64

    config.num_ep = 40

    config_ranges = OrderedDict()
    config_ranges['rnn_type'] = ['plainRNN',
                                 'CTRNN',
                                 'LSTM',
                                 'GRU',
                                 'RNNSTSP']
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def timecode():
    config = BaseConfig()
    config.experiment_name = 'timecode'
    config.rnn_type = 'plainRNN'
    config.t_scale = 1.0
    config.augment = None
    config.use_velocity = False
    config.context = 'zero'
    config.context_w = 10
    config.hidden_size = 64

    config.num_ep = 40

    config_ranges = OrderedDict()
    config_ranges['context'] = ['zero', 'noise', 'scalar', 'ramping',
                                'clock', 'stairs_end', 'stairs_start']
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def timecode_aug():
    config = BaseConfig()
    config.experiment_name = 'timecode_aug'
    config.rnn_type = 'plainRNN'
    config.t_scale = 1.0
    config.augment = (0.5, 1.5)
    config.use_velocity = False
    config.context = 'zero'
    config.context_w = 10
    config.hidden_size = 64

    config.num_ep = 40

    config_ranges = OrderedDict()
    config_ranges['context'] = ['zero', 'noise', 'scalar', 'ramping',
                                'clock', 'stairs_end', 'stairs_start']
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


# -----------------------------------------------------
# analysis
# -----------------------------------------------------

def timescale_analysis():
    configs = timescale()
    init_analysis(configs)
    t_scale_list = np.arange(0.1, 2, 0.1)
    acc_list = np.zeros_like(t_scale_list)
    for cfg in configs:
        for i_s, t_scale in enumerate(t_scale_list):
            new_cfg = copy.deepcopy(cfg)
            new_cfg.t_scale = t_scale
            acc_list[i_s] = evaluate.eval_total_acc(new_cfg)

        np.save(os.path.join(cfg.save_path, 'tscalelist.npy'), t_scale_list)
        np.save(os.path.join(cfg.save_path, 'acclist.npy'), acc_list)
        plots.plot_gen(t_scale_list, acc_list, cfg.rnn_type)
    plots.plot_group_gen(configs, configs[0].experiment_name, mode='rnn_type')


def timescale_aug_analysis():
    configs = timescale_aug()
    init_analysis(configs)
    t_scale_list = np.arange(0.1, 2, 0.1)
    acc_list = np.zeros_like(t_scale_list)
    for cfg in configs:
        for i_s, t_scale in enumerate(t_scale_list):
            new_cfg = copy.deepcopy(cfg)
            new_cfg.t_scale = t_scale
            acc_list[i_s] = evaluate.eval_total_acc(new_cfg)

        np.save(os.path.join(cfg.save_path, 'tscalelist.npy'), t_scale_list)
        np.save(os.path.join(cfg.save_path, 'acclist.npy'), acc_list)
        plots.plot_gen(t_scale_list, acc_list, cfg.rnn_type)
    plots.plot_group_gen(configs, configs[0].experiment_name, mode='rnn_type')


def timecode_analysis():
    configs = timecode()
    init_analysis(configs)
    t_scale_list = np.arange(0.1, 2, 0.1)
    acc_list = np.zeros_like(t_scale_list)
    for cfg in configs:
        for i_s, t_scale in enumerate(t_scale_list):
            new_cfg = copy.deepcopy(cfg)
            new_cfg.t_scale = t_scale
            acc_list[i_s] = evaluate.eval_total_acc(new_cfg)

        np.save(os.path.join(cfg.save_path, 'tscalelist.npy'), t_scale_list)
        np.save(os.path.join(cfg.save_path, 'acclist.npy'), acc_list)
        plots.plot_gen(t_scale_list, acc_list, cfg.context)
    plots.plot_group_gen(configs, configs[0].experiment_name, mode='context')


def timecode_aug_analysis():
    configs = timecode_aug()
    init_analysis(configs)
    t_scale_list = np.arange(0.1, 2, 0.1)
    acc_list = np.zeros_like(t_scale_list)
    for cfg in configs:
        for i_s, t_scale in enumerate(t_scale_list):
            new_cfg = copy.deepcopy(cfg)
            new_cfg.t_scale = t_scale
            acc_list[i_s] = evaluate.eval_total_acc(new_cfg)

        np.save(os.path.join(cfg.save_path, 'tscalelist.npy'), t_scale_list)
        np.save(os.path.join(cfg.save_path, 'acclist.npy'), acc_list)
        plots.plot_gen(t_scale_list, acc_list, cfg.context)
    plots.plot_group_gen(configs, configs[0].experiment_name, mode='context')
