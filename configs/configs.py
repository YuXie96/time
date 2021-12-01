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
        self.grad_clip = 1.0
        self.readout_steps = 1
        self.batch_s = 20
        self.use_velocity = True
        self.num_ep = 20
        self.max_batch = 30000

        self.save_every = 100
        self.log_every = 100
        self.early_stop = False
        self.eslen = 5

        # basic training parameters

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)