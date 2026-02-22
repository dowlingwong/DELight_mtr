import os
import sys

import numpy as np

from scipy.fft import *
from scipy.interpolate import interp1d
from scipy.integrate import quad

class NoiseGenerator(object):
    
    def __init__(self, config):

        self.set_config(config)
        
        
    def set_config(self, config):
        
        self._expected_files = ['noise_type', 'noise_power', 'sampling_frequency']
        for k in self._expected_files:
            if not k in config.keys():
                raise RuntimeError(f"Configuration field {k} not found in configuration dictionary.")
                
        self._set_spectra()
        self.set_noise_type(config['noise_type'])
        self.set_noise_power(config['noise_power'])
        self.sampling_frequency = config['sampling_frequency']
        
        
    def set_noise_type(self, noise_type):
        
        if noise_type.lower() in 'white blue violet brownian pink'.split():
            self.noise_type = noise_type.lower()
        else:
            if os.path.isfile(noise_type):
                self.noise_path = os.path.abspath(noise_type)
                self.noise_type = 'custom'
                self._load_psd()
            else:
                raise RuntimeError(f"Configuration noise_type field {noise_type} is neither a noise type nor a path.")
        self.spectrum = self._spectra[self.noise_type]
        

    def set_noise_power(self, noise_power):
        
        self.psd_area = noise_power
                
                
    def _set_spectra(self):
        
        self._spectra = {'white': lambda f: np.ones(len(f)),
                         'blue': lambda f: f,
                         'violet': lambda f: f**2,
                         'brownian': lambda f: 1/np.where(f == 0, float('inf'), f**2),
                         'pink': lambda f: 1/np.where(f == 0, float('inf'), f)}
        self._normalize = {'white': lambda f: 1. / (np.max(f) - np.min(f)),
                           'blue': lambda f: 2. / (np.max(f)**2 - np.min(f)**2),
                           'violet': lambda f: 3. / (np.max(f)**3 - np.min(f)**3),
                           'brownian': lambda f: 1. / (1. / np.sort(f)[1] - 1. / np.max(f)),
                           'pink': lambda f: 1. / (np.log(np.max(f)) - np.log(np.sort(f)[1]))}
        
    
    def _load_psd(self):
        
        self.noise_psd_data = np.load(self.noise_path)
        self._spectra['custom'] = interp1d(self.noise_psd_data[0], self.noise_psd_data[1], 
                                           fill_value=(self.noise_psd_data[1][0], self.noise_psd_data[1][-1]))
        self._normalize['custom'] = lambda f: np.ones_like(f)
        
        
    def generate_noise(self, N):
        
        frequencies = rfftfreq(N, d=1./self.sampling_frequency)
        psd_shape = self.spectrum(frequencies)
        # the psd is normalized by the frequency bin, thus the normalization factor is N / f_s, but the 
        # fft is multiplied by N, so the psd (which is the amplitude squared of the fft) has a N² factor
        # that is to be normalized and this means that N / f_s * 1. / N² = 1. / N / f_s. This is the 
        # normalization of the given PSD
        if self.noise_type == 'custom':
            if len(frequencies) > 1:
                current_power = np.trapz(psd_shape, frequencies)
            scale = self.psd_area / current_power
            psd = scale * psd_shape * self.sampling_frequency * N
        else:
            norm = (
                0.5
                * self.psd_area
                * self._normalize[self.noise_type](frequencies)
                * self.sampling_frequency
                * N
            )
            psd = norm * psd_shape
            
        psd_amp = np.sqrt(psd)
        phi = np.random.uniform(0, 2 * np.pi, len(psd))  
        x_psd = psd_amp * np.exp(1j * phi)
        y_psd = irfft(x_psd, n=N)
        return y_psd
    