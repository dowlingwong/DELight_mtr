#Optimum filter V1.1, adapt to the case where the trace length is longer than the template
import numpy as np
from numpy.fft import rfft, irfft, fft, ifft, fftfreq, rfftfreq
from scipy.signal import fftconvolve


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
        self._filter_kernel = self._kernel_fft / self._kernel_normalization
        self._kernel_td = np.real(ifft(self._filter_kernel)) * self._sampling_frequency
        
        
    def fit(self, trace):
        trace_fft = fft(trace, axis=-1)/self._sampling_frequency # V
        trace_filtered = self._filter_kernel * trace_fft
        amp = np.real(trace_filtered.sum(axis=-1)) * self._sampling_frequency / self._length
        chisq0 = np.real((trace_fft.conj() * trace_fft / self._noise_psd_unfolded).sum()) * self._sampling_frequency / self._length
        chisq = (chisq0 - amp**2 * self._kernel_normalization) / (self._length - 2) 
        trace_filtered_td = np.real(
            ifft(trace_filtered, axis=-1)
        ) * self._sampling_frequency
        return amp, chisq
    
    def fit_with_shift(self, trace, allowed_shift_range=None):
 
        trace_fft = fft(trace, axis=-1)/self._sampling_frequency # V
        trace_filtered = self._filter_kernel * trace_fft
        trace_filtered_td = np.real(
            ifft(trace_filtered, axis=-1)
        ) * self._sampling_frequency

        chi0 = np.real((trace_fft.conj() * trace_fft / self._noise_psd_unfolded).sum()) * self._sampling_frequency / self._length
        chit_withdelay = (trace_filtered_td ** 2) * self._kernel_normalization
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

    def rolling_fit_with_shift(self, trace_long, allowed_shift_range=None):
        L = len(trace_long)
        N = self._length
        fs = self._sampling_frequency

        n_windows = L // N
        amps = np.zeros(n_windows)
        chisqs = np.zeros(n_windows)
        shifts = np.zeros(n_windows, dtype=int)
        positions = np.zeros(n_windows, dtype=int)

        for i in range(n_windows):
            start = i * N
            segment = trace_long[start:start + N]
            amp, chisq, t0 = self.fit_with_shift(segment, allowed_shift_range)

            amps[i] = amp
            chisqs[i] = chisq
            shifts[i] = t0
            positions[i] = start + t0

        return amps, chisqs, shifts, positions
    
    
    def convolve_long_trace(self, trace_long):

        filtered_trace = fftconvolve(trace_long, self._kernel_td[::-1], mode='valid')
        return filtered_trace
    
    def padding_OF(self, raw_trace):
        """
        Apply optimum filter to a raw trace by padding the internal template to match trace length.

        Parameters:
        - raw_trace (1D array): The trace to filter.

        Returns:
        - amplitude (float): Best-fit amplitude at the delay that minimizes chi².
        """
        L_example = len(raw_trace)
        L_template = len(self._template)
        
        # Pad the template to the trace length
        template_padded = np.pad(self._template, (0, L_example - L_template))
        template_fft = fft(template_padded) / self._sampling_frequency
        example_fft = fft(raw_trace) / self._sampling_frequency

        # Recompute unfolded PSD to match trace length
        if L_example % 2 == 0:
            noise_psd_unfolded = np.concatenate((
                [np.inf],
                self._noise_psd[1:-1] / 2,
                [self._noise_psd[-1]],
                self._noise_psd[-2:0:-1] / 2
            ))
        else:
            noise_psd_unfolded = np.concatenate((
                [np.inf],
                self._noise_psd[1:] / 2,
                self._noise_psd[-1:0:-1] / 2
            ))

        noise_psd_unfolded = np.pad(noise_psd_unfolded, (0, L_example - len(noise_psd_unfolded)))
        noise_psd_unfolded = np.where(noise_psd_unfolded == 0, np.inf, noise_psd_unfolded)

        # Filter kernel construction
        kernel_fft = template_fft.conjugate() / noise_psd_unfolded
        kernel_normalization = np.real(np.dot(kernel_fft, template_fft)) * self._sampling_frequency / L_example
        filter_kernel_fft = kernel_fft / kernel_normalization

        # Apply filter
        filtered_fft = example_fft * filter_kernel_fft
        amp_t0_global = np.real(ifft(filtered_fft)) * self._sampling_frequency

        # Compute chi²
        power_fft = example_fft.conj() * example_fft / noise_psd_unfolded
        chi0_global = np.real(np.sum(power_fft)) * self._sampling_frequency / L_example
        chit_withdelay = amp_t0_global**2 * kernel_normalization
        chi2_t0_global = (chi0_global - chit_withdelay) / (L_example - 2)

        # Get amplitude at minimum chi²
        best_index = int(np.argmin(chi2_t0_global))
        best_amplitude = amp_t0_global[best_index]

        return best_amplitude
