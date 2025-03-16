from .RQCalculator import *
import numpy as np
import yaml

class TimeShift(RQCalculator):

    __version__ = '0.0.1'
    data_type = [('time_shift', np.uint16)]

    def __init__(self, config_file):
        self.load_config(config_file)

    def apply(self, traces, rqs):

        def get_timeshift(trace):
            shift = np.argmax(np.diff(trace, n=2)>5*np.std(np.diff(trace, n=2)))
            if shift == 0:
                shift = np.argmax(np.diff(trace, n=2)>4*np.std(np.diff(trace, n=2)))
            return shift      
        
        def get_timeshifts():
            time_shifts = np.zeros(traces.shape[:-1])
            for id, event in enumerate(traces):
                    for channel, trace in enumerate(event):
                        time_shifts[id][channel] = get_timeshift(trace)
            return time_shifts
        

        rqs['time_shift'] = get_timeshifts()


    def load_config(self, config_file):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
        return
