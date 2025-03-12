from .RQCalculator import *
import numpy as np
import yaml

class RiseTime(RQCalculator):

    __version__ = '0.0.1'
    data_type = [('rise_time', np.float64)]
    dependencies = ['Amplitude', 'MeanBaseline']

    def __init__(self, config_file):
        self.load_config(config_file)

    def apply(self, traces, rqs):

        def get_rising_time(trace, low, high, sample_time, baseline, amplitude):
            """returns rising time in seconds"""
            low_arg = np.argmax(trace[:np.argmax(trace)]
                                 - baseline > low*amplitude)
            high_arg = np.argmax(trace[:np.argmax(trace)]
                                 - baseline > high*amplitude)
            return (high_arg-low_arg) * sample_time

        def get_rising_times(low, high, sample_time):
            rising_time = np.zeros(traces.shape[:-1])
            for id, event in enumerate(traces):
                    for channel, trace in enumerate(event):
                        baseline = rqs['mean_baseline'][id][channel]
                        amplitude = rqs['A'][id][channel]
                        rising_time[id][channel] = get_rising_time(trace, 
                                                    low, 
                                                    high,
                                                    sample_time,
                                                    baseline,
                                                    amplitude)
            return rising_time
        
        low_perc = self.config.get('low', 0.1)
        high_perc = self.config.get('high', 0.9)
        sample_time = self.config_glob['trigger_information']['time_per_sample']

        rqs['rise_time'] = get_rising_times(low_perc, high_perc, sample_time)


    def load_config(self, config_file):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
        with open('./config.yaml') as f:
            self.config_glob = yaml.safe_load(f)
        return
