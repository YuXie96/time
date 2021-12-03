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
    config.t_scale = 1.0
    config.use_velocity = False

    config_ranges = OrderedDict()
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
    cfg = configs[0]
    for i_s, t_scale in enumerate(t_scale_list):
        new_cfg = copy.deepcopy(cfg)
        new_cfg.t_scale = t_scale
        acc_list[i_s] = evaluate.eval_total_acc(new_cfg)

    np.save(os.path.join(cfg.save_path, 'tscalelist.npy'), t_scale_list)
    np.save(os.path.join(cfg.save_path, 'acclist.npy'), acc_list)
    plots.plot_gen(t_scale_list, acc_list)
