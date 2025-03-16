from .RQCalculator import *
import numpy as np
import yaml
from scipy.optimize import curve_fit, minimize

class TemplateFitting(RQCalculator):

    __version__ = '0.0.1'
    data_type = [('A_templ', np.float64), ('chi2_templ', np.float64), ('baseline_templ', np.float64)]
    dependencies = ['MeanBaseline']

    def __init__(self, config_file):
        self.load_config(config_file)

    def apply(self, traces, rqs):

        def load_template():
            try:
                templ_path = self.config['template_path']
                template = np.load(templ_path)
            except KeyError:
                template = np.ones(traces.shape[-1])
            
            return template
        

        def chi_square_reduced(params, trace, template, sigma):
            A, O = params
            dof = (len(trace)-len(params)) * sigma**2
            return 1/dof * np.sum((trace - A*template - O)**2) 
        
        
        def fit_template(trace, template, sigma, A_guess=10000.0, O_guess=0.0):
            if 'start_values' in self.config:
                A_guess, O_guess = self.config['start_values']
            return minimize(chi_square_reduced, [A_guess, O_guess], args=(trace, template, sigma))

    
        def fit_templates(template, traces):
            chi2s = np.zeros(traces.shape[:-1])
            As = np.zeros(traces.shape[:-1])
            Os = np.zeros(traces.shape[:-1])

            for id, event in enumerate(traces):
                for channel, trace in enumerate(event):
                    baseline_mean = rqs['mean_baseline'][id][channel]
                    baseline_std = rqs['std_baseline'][id][channel]
                    trace -= baseline_mean
                    fit_result = fit_template(trace, template, baseline_std)
                    chi2s[id][channel] = fit_result.fun
                    As[id][channel] = fit_result.x[0]
                    Os[id][channel] = fit_result.x[1]

            return chi2s, As, Os

        
        template = load_template()
        chi2s, As, Os = fit_templates(template, traces)

        rqs['A_templ'] = As
        rqs['baseline_templ'] = Os
        rqs['chi2_templ'] = chi2s


    def load_config(self, config_file):
        with open(config_file) as f:
            self.config = yaml.safe_load(f)

        return
