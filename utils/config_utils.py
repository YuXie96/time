import logging
import os
import json
from copy import deepcopy

import numpy as np

from configs.config_global import ROOT_DIR


def auto_name(configs, config_diffs):
    """Helper function for automatically naming models based on configs."""
    new_configs = list()
    for config, config_diff in zip(configs, config_diffs):
        name = 'model'
        for key, val in config_diff.items():
            name += '_' + str(key) + str(val)
        # replace char that are not suitable for path
        name = name.replace(",", "").replace(" ", "_")
        name = name.replace("[", "_").replace("]", "_")
        config.model_name = name
        config.save_path = os.path.join(ROOT_DIR, 'experiments',
                                        config.experiment_name, config.model_name)
        new_configs.append(config)
    return new_configs


def save_config(config, save_path, also_save_as_text=True):
    """
    Save config.
    adapted from https://github.com/gyyang/olfaction_evolution
    """
    if os.path.exists(save_path):
        logging.warning('Save dir {} already exists!'.format(save_path)
                        + 'Storing info there anyway.')
    else:
        os.makedirs(save_path)

    config_dict = config.__dict__
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f)

    if also_save_as_text:
        with open(os.path.join(save_path, 'config.txt'), "w") as f:
            for k, v in config_dict.items():
                f.write(str(k) + ' >>> ' + str(v) + '\n\n')


def load_config(save_path):
    """
    Load config.
    adapted from https://github.com/gyyang/olfaction_evolution
    """
    import configs.configs
    with open(os.path.join(save_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    config = configs.configs.BaseConfig()
    for key, val in config_dict.items():
        setattr(config, key, val)
    return config


def _vary_config_combinatorial(base_config, config_ranges):
    """Return combinatorial configurations.
    adapted from https://github.com/gyyang/olfaction_evolution

    Args:
        base_config: BaseConfig object, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }
            config_ranges should not have repetitive values

    Return:
        configs: a list of config objects [config1, config2, ...]
            Loops over all possible combinations of hp1, hp2, ...
        config_diffs: a list of config diff from base_config
    """
    # Unravel the input index
    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]
    n_max = int(np.prod(dims))

    configs, config_diffs = list(), list()
    for i in range(n_max):
        new_config = deepcopy(base_config)

        config_diff = dict()
        indices = np.unravel_index(i, shape=dims)
        # Set up new config
        for key, index in zip(keys, indices):
            val = config_ranges[key][index]
            setattr(new_config, key, val)
            config_diff[key] = val

        configs.append(new_config)
        config_diffs.append(config_diff)

    return configs, config_diffs


def _vary_config_sequential(base_config, config_ranges):
    """Return sequential configurations.
    adapted from https://github.com/gyyang/olfaction_evolution

    Args:
        base_config: BaseConfig object, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }
            config_ranges values should have equal length,
            otherwise this will only loop through the shortest one

    Return:
        configs: a list of config objects [config1, config2, ...]
            Loops over all hyperparameters hp1, hp2 together sequentially
        config_diffs: a list of config diff from base_config
    """
    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]
    if len(dims) == 0:
        n_max = 1
    else:
        n_max = np.min(dims)

    configs, config_diffs = list(), list()
    for i in range(n_max):
        new_config = deepcopy(base_config)
        config_diff = dict()
        for key in keys:
            val = config_ranges[key][i]
            setattr(new_config, key, val)
            config_diff[key] = val

        configs.append(new_config)
        config_diffs.append(config_diff)

    return configs, config_diffs


def _vary_config_control(base_config, config_ranges):
    """Return control configurations.
    adapted from https://github.com/gyyang/olfaction_evolution

    Each config_range is gone through sequentially. The base_config is
    trained only once.

    Args:
        base_config: BaseConfig object, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }
            base_config must contain keys in config_ranges

    Return:
        configs: a list of config objects [config1, config2, ...]
            Loops over all hyperparameters hp1, hp2 independently
        config_diffs: a list of config diff from base_config
    """
    keys = list(config_ranges.keys())
    # Remove the base_config value from the config_ranges
    new_config_ranges = {}
    for key, val in config_ranges.items():
        base_config_val = getattr(base_config, key)
        new_config_ranges[key] = [v for v in val if v != base_config_val]

    # Unravel the input index
    dims = [len(new_config_ranges[k]) for k in keys]
    n_max = int(np.sum(dims))

    configs, config_diffs = list(), list()
    configs.append(deepcopy(base_config))
    config_diffs.append({})

    for i in range(n_max):
        new_config = deepcopy(base_config)

        index = i
        for j, dim in enumerate(dims):
            if index >= dim:
                index -= dim
            else:
                break

        config_diff = dict()
        key = keys[j]

        val = new_config_ranges[key][index]
        setattr(new_config, key, val)
        config_diff[key] = val

        configs.append(new_config)
        config_diffs.append(config_diff)

    return configs, config_diffs


def vary_config(base_config, config_ranges, mode, default_name=False):
    """Return configurations.
    adapted from https://github.com/gyyang/olfaction_evolution

    Args:
        base_config: BaseConfig object, a base configuration
        config_ranges: a dictionary of hyperparameters values
            config_ranges = {
                'hp1': [hp1_val1, hp1_val2, ...],
                'hp2': [hp2_val1, hp2_val2, ...],
            }
        mode: str, can take 'combinatorial', 'sequential', and 'control'
        default_name: bool, whether to use auto_name function

    Return:
        configs: a list of config objects [config1, config2, ...]
    """
    if mode == 'combinatorial':
        _vary_config = _vary_config_combinatorial
    elif mode == 'sequential':
        _vary_config = _vary_config_sequential
    elif mode == 'control':
        _vary_config = _vary_config_control
    else:
        raise ValueError('Unknown mode {}'.format(str(mode)))
    configs, config_diffs = _vary_config(base_config, config_ranges)
    if default_name:
        for i, config in enumerate(configs):
            config.model_name = str(i).zfill(6)  # default name
            config.save_path = os.path.join(ROOT_DIR, 'experiments',
                                            config.experiment_name,
                                            config.model_name)
    else:
        # Automatic set names for configs
        configs = auto_name(configs, config_diffs)
    return configs
