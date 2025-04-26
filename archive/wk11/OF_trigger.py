import numpy as np
from numpy.fft import rfft, irfft, fft, ifft, fftfreq, rfftfreq

# Optimization ideas:

# 1. The _update_state() function can be split into update_template and update_psd, to save time in cases
# when we only want to change the PSD, keeping the template the same. It would save us a couple FFTs

# 2. We are dealing with real-valued traces. We can use folded PSDs and rfft/irfft.
# This would make all the FFTs more than 2 times faster. Special care should be taken with the PSD edges.

class OptimumFilter():
    
    def __init__(self, template, noise_psd, sampling_frequency):
        self._template = template
        self._noise_psd = noise_psd
        self._sampling_frequency = sampling_frequency
        self._update_state()
        
    def set_template(self, template):
        self._template = template
        self._update_state()
        
    def set_noise_psd(self, noise_psd):
        self._noise_psd = noise_psd
        self._update_state()
        
    def _update_state(self):
        self._length = len(self._template)
        
        if self._length%2==0:
            self._noise_psd_unfolded = np.concatenate(([np.inf],
                                                       self._noise_psd[1:-1]/2,
                                                       [self._noise_psd[-1]],
                                                       self._noise_psd[-2:0:-1]/2))
        else:
            self._noise_psd_unfolded = np.concatenate(([np.inf],
                                                       self._noise_psd[1:]/2,
                                                       self._noise_psd[-1:0:-1]/2))
            
        
        self._template_fft = fft(self._template)/self._sampling_frequency
        
        self._kernel_fft = self._template_fft.conjugate() / self._noise_psd_unfolded
        self._kernel_normalization = np.real(np.dot(self._kernel_fft, self._template_fft))*self._sampling_frequency/self._length 
        
    def fit(self, trace):#fit only one trace
        trace_fft = fft(trace, axis=-1)/self._sampling_frequency # V
        trace_filtered = self._kernel_fft * trace_fft / self._kernel_normalization # Phi * V
        chisq0 = np.real(
            np.dot(trace_fft.conjugate()/self._noise_psd_unfolded,
                   trace_fft)*self._sampling_frequency/self._length
        )
        
        amp0 = np.real(np.sum(
            trace_filtered, axis=-1
        ))*self._sampling_frequency/self._length


        # total chisq
        # self._length-2 is the assumed number of degrees of freedom
        chisq = (chisq0 - (amp0**2)*self._kernel_normalization)/(self._length-2)
        
        return amp0, chisq
        
    def fit_with_shift(self, trace, allowed_shift_range=None):
 
        trace_fft = fft(trace, axis=-1)/self._sampling_frequency # V
        trace_filtered = self._kernel_fft * trace_fft / self._kernel_normalization # Phi * V
        trace_filtered_td = np.real(
            ifft(trace_filtered, axis=-1)
        ) * self._sampling_frequency

        chi0 = np.real(
            np.dot(trace_fft.conjugate()/self._noise_psd_unfolded,
                   trace_fft)*self._sampling_frequency/self._length
        )

        chit_withdelay = (trace_filtered_td**2) * self._kernel_normalization
        chi = chi0 - chit_withdelay
        
        if allowed_shift_range is None:
            ind = np.arange(len(chi))
        else:
            ind = np.concatenate((np.arange(self._length+allowed_shift_range[0], self._length),
                                  np.arange(allowed_shift_range[1]+1)))
        
        best_ind = ind[np.argmin(chi[ind], axis=-1)]
        
        amp = trace_filtered_td[best_ind]
        chisq = chi[best_ind]/(self._length-3)
        t0 = best_ind if best_ind<self._length//2 else best_ind-self._length
        
        return amp, chisq, t0