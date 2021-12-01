"""
Configurations for the project
format adapted from https://github.com/gyyang/olfaction_evolution
"""


class BaseConfig(object):
    def __init__(self):
        """
        model_type: model type, eg. "ConvRNNBL"
        task_type: task type, eg. "n_back"
        """
        self.experiment_name = None
        self.model_name = None
        self.save_path = None

        # max norm of grad clipping, eg. 1.0 or None
        self.grad_clip = None

        # basic training parameters

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)