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
from collections import OrderedDict
import logging

from configs.config_global import ROOT_DIR, LOG_LEVEL
from configs.configs import BaseConfig
from utils.config_utils import vary_config

from analysis.train_analysis import plot_train_log
from evaluate import eval_class_acc


def init_analysis(configs_):
    logging.basicConfig(level=LOG_LEVEL)
    exp_name = configs_[0].experiment_name
    print('Analyzing ' + exp_name)
    exp_path = os.path.join(ROOT_DIR, 'experiments', exp_name) + os.sep
    plot_train_log([exp_path], exp_name=exp_name)


# -----------------------------------------------------
# experiments
# -----------------------------------------------------


def test():
    config = BaseConfig()
    config.experiment_name = 'test'

    config_ranges = OrderedDict()
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


# -----------------------------------------------------
# analysis
# -----------------------------------------------------

def test_analysis():
    configs = test()
    init_analysis(configs)

    for cfg in configs:
        eval_class_acc(cfg)
