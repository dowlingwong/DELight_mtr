#Optimum filter V1, adapt to the case where the trace length is longer than the template
import numpy as np
from numpy.fft import rfft, irfft, fft, ifft, fftfreq, rfftfreq

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
        
    def fit(self, trace):

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
    
    def fit_with_shift(self, trace, allowed_shift_range=[-2000, 2000]):
 
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

    def rolling_fit_0(self, trace_long, allowed_shift_range=None):
        L = len(trace_long)
        N = self._length  # Length of the template/filter (e.g., 32768)

        n_windows = L // N  # number of non-overlapping windows

        amps = np.zeros(n_windows)
        chisqs = np.zeros(n_windows)
        shifts = np.zeros(n_windows, dtype=int)
        positions = np.zeros(n_windows, dtype=int)

        for i in range(n_windows):
            start = i * N
            segment = trace_long[start : start + N]
            amp, chisq, t0 = self.fit_with_shift(segment, allowed_shift_range)

            amps[i] = amp
            chisqs[i] = chisq
            shifts[i] = t0
            positions[i] = start + t0  # corrected absolute shift for non-overlapping window

        return amps, chisqs, shifts, positions
    
    def rolling_fit_1(self, trace_long, step=1):
        N = self._length  # Fixed window size = template length

        n_windows = (len(trace_long) - N) // step + 1  # total number of valid windows

        amps = np.zeros(n_windows)
        chisqs = np.zeros(n_windows)
        shifts = np.zeros(n_windows, dtype=int)  # renamed from positions

        for i in range(n_windows):
            start = i * step
            end = start + N
            segment = trace_long[start:end]

            amp, chisq = self.fit(segment)

            amps[i] = amp
            chisqs[i] = chisq
            shifts[i] = start  # beginning of the window

        return amps, chisqs, shifts

