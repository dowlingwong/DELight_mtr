from .RQCalculator import *
import numpy as np
import yaml

class MeanBaseline(RQCalculator):

    __version__ = '0.0.1'
    data_type = [('mean_baseline', np.float64),('std_baseline', np.float64)]

    def __init__(self, config_file):

        self.load_config(config_file)

    def apply(self, traces, rqs):
        pre_trig_bins = self.config_glob['trigger_information']['pre-trigger_samples']
        rqs['mean_baseline'] = np.mean(traces[:,:,:pre_trig_bins],axis=2)
        rqs['std_baseline'] = np.std(traces[:,:,:pre_trig_bins],axis=2)

    def load_config(self, config_file):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
        with open('./config.yaml') as f:
            self.config_glob = yaml.safe_load(f)
        return
