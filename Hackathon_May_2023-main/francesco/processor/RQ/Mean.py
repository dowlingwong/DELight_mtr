from .RQCalculator import *
import numpy as np

class Mean(RQCalculator):

    __version__ = '0.0.1'
    data_type = [('mean', np.float64)]

    def __init__(self, config_file):

        self.load_config(config_file)

    def apply(self, traces, rqs):

        rqs['mean'] = np.mean(traces, axis=-1)

    def load_config(self, config_file):
        self.config = None
        return
