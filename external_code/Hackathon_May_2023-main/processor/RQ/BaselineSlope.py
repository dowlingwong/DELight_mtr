from .RQCalculator import *
import numpy as np
import yaml

class BaselineSlope(RQCalculator):

    __version__ = '0.0.1'
    data_type = [('baseline_slope', np.float64)]

    def __init__(self, config_file):
        self.load_config(config_file)

    def apply(self, traces, rqs):

        def get_baseline_slope(trace):
            """returns baseline slope for each trace in ADCcounts per time bins"""
            x = np.arange(0, len(trace))
            slope, yoffset = np.polyfit(x, trace, deg=1)

            return slope

        def get_baseline_slopes():

            baseline_slope = np.zeros(traces.shape[:-1])
            for id, event in enumerate(traces):
                    for channel, trace in enumerate(event):
                        baseline = trace[:pretrigger_samples]
                        baseline_slope[id][channel] = get_baseline_slope(baseline)
            return baseline_slope
        
        pretrigger_samples = self.config_glob['trigger_information']['pre-trigger_samples']

        rqs['baseline_slope'] = get_baseline_slopes()

    def load_config(self, config_file):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
        with open('./config.yaml') as f:
            self.config_glob = yaml.safe_load(f)
        return
