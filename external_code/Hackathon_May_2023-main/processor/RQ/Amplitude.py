from .RQCalculator import *
import numpy as np
import yaml

class Amplitude(RQCalculator):

    __version__ = '0.0.1'
    data_type = [('A', np.float64)]
    dependencies = ['MeanBaseline']


    def __init__(self, config_file):
        self.load_config(config_file)


    def apply(self, traces, rqs):
        rqs['A'] = np.max(traces, axis=-1) - rqs['mean_baseline']


    def load_config(self, config_file):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
        return
