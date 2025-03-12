from .RQCalculator import *
import numpy as np

class StandardDeviation(RQCalculator):

    data_type = [('std', np.float64, 'Standard deviation')]

    def apply(self, traces, rqs):

        return np.std(traces, axis=-1)

    def load_config(self, config_file):

        return
