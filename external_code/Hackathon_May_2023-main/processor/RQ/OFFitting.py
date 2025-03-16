from .RQCalculator import *
from .OptimumFilter import *
import numpy as np
import yaml

class OFFitting(RQCalculator):

    __version__ = '0.0.1'
    data_type = [('OF_ampl', np.float64), ('OF_chi2', np.float64), ('OF_time', int)]

    def __init__(self, config_file):
        self.load_config(config_file)

    def apply(self, traces, rqs):    
        s = traces.shape
        if self.config['with_shift']:
            fit_results = np.array([self.of_fitter.fit_with_shift(t, allowed_shift_range=self.config['allowed_shift_range']) for t in traces.reshape((s[0]*s[1],s[2]))])
            rqs['OF_ampl'] = fit_results[:,0].reshape((s[0],s[1]))
            rqs['OF_chi2'] = fit_results[:,1].reshape((s[0],s[1]))
            rqs['OF_time'] = fit_results[:,2].reshape((s[0],s[1]))
        else:
            fit_results = np.array([self.of_fitter.fit(t) for t in traces.reshape((s[0]*s[1],s[2]))])
            rqs['OF_ampl'] = fit_results[:,0].reshape((s[0],s[1]))
            rqs['OF_chi2'] = fit_results[:,1].reshape((s[0],s[1]))
    
    def load_config(self, config_file):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)
            
        self.template = np.load(self.config['template_path'])
        self.psd = np.load(self.config['psd_path'])
        
        self.of_fitter = OptimumFilter(self.template, self.psd, self.config['sampling_frequency'])
        
        if self.config['with_shift']:
            self.data_type = [('OF_ampl', np.float64), ('OF_chi2', np.float64), ('OF_time', int)]
        else:
            self.data_type = [('OF_ampl', np.float64), ('OF_chi2', np.float64)]
            
