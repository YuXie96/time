import os
import subprocess
import argparse
import logging

import configs.experiments as experiments
from utils.config_utils import save_config
from train import model_train
from configs.config_global import LOG_LEVEL
from configs.configs import BaseConfig


def train_cmd(config_):
    arg = '\'' + config_.save_path + '\''
    command = r'''python -c "import train; train.train_from_path(''' + arg + ''')"'''
    return command


def analysis_cmd(exp_name):
    arg = exp_name + '_analysis()'
    command = r'''python -c "import configs.experiments as experiments; experiments.''' + arg + '''"'''
    return command


def get_jobfile(cmd, job_name, dep_ids=[], email=False,
                sbatch_path='./sbatch/', hours=8):
    """
    Create a job file.
    adapted from https://github.com/gyyang/olfaction_evolution

    Args:
        cmd: python command to be execute by the cluster
        job_name: str, name of the job file
        dep_ids: list, a list of job ids used for job dependency
        email: bool, whether or to send email about job status
        sbatch_path : str, Directory to store SBATCH file in.
        hours : int, number of hours to train
    Returns:
        job_file : str, Path to the job file.
    """
    assert type(dep_ids) is list, 'dependency ids must be list'
    assert all(type(id_) is str for id_ in dep_ids), 'dependency ids must all be strings'

    if len(dep_ids) == 0:
        dependency_line = ''
    else:
        dependency_line = '#SBATCH --dependency=afterok:' \
                          + ':'.join(dep_ids) + '\n'

    if not email:
        email_line = ''
    else:
        email_line = '#SBATCH --mail-type=ALL\n' + \
                     '#SBATCH --mail-user=rxie9596@outlook.com\n'

    os.makedirs(sbatch_path, exist_ok=True)
    job_file = os.path.join(sbatch_path, job_name + '.sh')

    with open(job_file, 'w') as f:
        f.write(
            '#!/bin/bash\n'
            + '#SBATCH -t {}:00:00\n'.format(hours)
            + '#SBATCH -N 1\n'
            + '#SBATCH -n 4\n'
            + '#SBATCH --mem=32G\n'
            + '#SBATCH --gres=gpu:1\n'
            + '#SBATCH --constraint=high-capacity\n'
            + '#SBATCH -p yanglab\n'
            + '#SBATCH -e ./sbatch/slurm-%j.out\n'
            + '#SBATCH -o ./sbatch/slurm-%j.out\n'
            + dependency_line
            + email_line
            + '\n'
            + 'module load openmind/cuda/10.2\n'
            + 'source ~/.bashrc\n'
            + 'conda activate time\n'
            + 'cd /om2/user/yu_xie/projects/time\n'
            + cmd + '\n'
            + '\n'
            )
        print(job_file)
    return job_file


def train_experiment(experiment, on_cluster, use_exp_array):
    """Train model across platforms given experiment name.
    adapted from https://github.com/gyyang/olfaction_evolution
    Args:
        experiment: str, name of experiment to be run
            must correspond to a function in experiments.py
        on_cluster: bool, whether to run experiments on cluster
        use_exp_array: use dependency between training of different exps

    Returns:
        return_ids: list, a list of job ids that are last in the dependence
            if not using cluster, return is None
    """
    print('Training {:s} experiment'.format(experiment))
    if experiment in dir(experiments):
        # Get list of configurations from experiment function
        exp_configs = getattr(experiments, experiment)()
    else:
        raise ValueError('Experiment config not found: ', experiment)

    return_ids = []
    if not use_exp_array:
        # exp_configs is a list of configs
        assert isinstance(exp_configs[0], BaseConfig), \
            'exp_configs should be list of configs'

        if on_cluster:
            for config in exp_configs:
                save_config(config, config.save_path)
                python_cmd = train_cmd(config)
                job_n = config.experiment_name + '_' + config.model_name
                cp_process = subprocess.run(['sbatch', get_jobfile(python_cmd,
                                                                   job_n,
                                                                   hours=config.hours)],
                                            capture_output=True, check=True)
                cp_stdout = cp_process.stdout.decode()
                print(cp_stdout)
                job_id = cp_stdout[-9:-1]
                return_ids.append(job_id)
        else:
            for config in exp_configs:
                save_config(config, config.save_path)
                model_train(config)
    else:
        # exp_configs is a list of lists of configs
        assert isinstance(exp_configs[0], list) \
               and isinstance(exp_configs[0][0], BaseConfig), \
               'exp_configs should a list of lists of configs'

        if on_cluster:
            # job_ids is a list (exp groups) of
            # list of job ids (different config in one exp)
            job_ids = []
            send_email = False
            for group_num, config_group in enumerate(exp_configs):
                job_ids.append([])
                for config in config_group:
                    save_config(config, config.save_path)
                    if group_num == 0:
                        pre_job_ids = []
                    else:
                        pre_job_ids = job_ids[group_num - 1]

                    if group_num == len(exp_configs) - 1:
                        send_email = True

                    python_cmd = train_cmd(config)
                    job_n = config.experiment_name + '_' + config.model_name
                    cp_process = subprocess.run(['sbatch',
                                                 get_jobfile(python_cmd, job_n,
                                                             dep_ids=pre_job_ids,
                                                             email=send_email,
                                                             hours=config.hours)],
                                                capture_output=True, check=True)
                    cp_stdout = cp_process.stdout.decode()
                    print(cp_stdout)
                    job_id = cp_stdout[-9:-1]
                    job_ids[group_num].append(job_id)

            return_ids = job_ids[-1]

        else:
            for config_group in exp_configs:
                for config in config_group:
                    save_config(config, config.save_path)
                    model_train(config)

    return return_ids


def analyze_experiment(experiment, prev_ids, on_cluster):
    """analyze experiments
     adapted from https://github.com/gyyang/olfaction_evolution

     Args:
         experiment: str, name of experiment to be analyzed
             must correspond to a function in experiments.py
         prev_ids: list, list of job ids
         on_cluster: bool, if use on the cluster
     """
    print('Analyzing {:s} experiment'.format(experiment))
    if (experiment + '_analysis') in dir(experiments):
        if on_cluster:
            python_cmd = analysis_cmd(experiment)
            job_n = experiment + '_analysis'
            slurm_cmd = ['sbatch', get_jobfile(python_cmd, job_n,
                                               dep_ids=prev_ids, email=True)]
            cp_process = subprocess.run(slurm_cmd, capture_output=True,
                                        check=True)
            cp_stdout = cp_process.stdout.decode()
            print(cp_stdout)
        else:
            getattr(experiments, experiment + '_analysis')()
    else:
        raise ValueError('Experiment analysis not found: ', experiment + '_analysis')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', nargs='+', help='Train experiments', default=[])
    parser.add_argument('-a', '--analyze', nargs='+', help='Analyze experiments', default=[])
    parser.add_argument('-c', '--cluster', action='store_true', help='Use batch submission on cluster')
    args = parser.parse_args()
    experiments2train = args.train
    experiments2analyze = args.analyze
    logging.basicConfig(level=LOG_LEVEL)
    # on openmind cluster
    use_cluster = args.cluster
    # use_cluster = 'node' in platform.node() or 'dgx' in platform.node()

    train_ids = []
    if experiments2train:
        for exp in experiments2train:
            exp_array = '_exp_array' in exp
            exp_ids = train_experiment(exp, on_cluster=use_cluster,
                                       use_exp_array=exp_array)
            train_ids += exp_ids

    if experiments2analyze:
        for exp in experiments2analyze:
            analyze_experiment(exp, prev_ids=train_ids,
                               on_cluster=use_cluster)
