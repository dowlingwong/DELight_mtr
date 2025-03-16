from .RQCalculator import *
import numpy as np
import yaml
#import OptimumFilter #import the base OF class. It needs a different name

class OptimumFilterRQCalculator(RQCalculator):
    
    def __init__(self):
        self._optimum_filter = OptimumFilter() # this is the base OF class

    #data_type = [('OF', np.float64, 'Standard deviation')]

    def apply(self, traces, rqs):
        a,chi2,x0 = self._optimum_filter.apply(trace)
        rqs['a'] = 1 #trace index??
        # chi2 and x0 goes here
        

    def load_config(self, config_file):
        with open(config_file) as f:
            config = yaml.safe_load(f)    
        self._optimum_filter.set_template(np.load(config['template']))
        self._optimum_filter.set_noise_psd(np.load(config['noise_psd']))
        
        
