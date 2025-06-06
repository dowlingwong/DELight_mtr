import numpy as np
import scipy.fft as sp_fft

class OFtrigger:

    def __init__(self, template, noise_psd, sampling_frequency):
        self._sampling_frequency = sampling_frequency
        self._length = len(template)
        self.set_template(template)
        self.set_noise_psd(noise_psd)

    def set_template(self, template):
        self._template = template
        self._template_fft = sp_fft.rfft(template) / self._sampling_frequency
        self._update_kernel_fft()

    def set_noise_psd(self, noise_psd):
        self._noise_psd = noise_psd.copy()

        self._inv_psd = np.zeros_like(noise_psd)
        self._inv_psd[1:] = 1.0 / (noise_psd[1:] + 1e-30)
        if self._length % 2 == 0:
            self._inv_psd[-1] = 1.0 / (noise_psd[-1] + 1e-30)
        self._inv_psd[0] = 0.0

        self._update_kernel_fft()

    def _update_kernel_fft(self):
        if hasattr(self, '_template_fft') and hasattr(self, '_inv_psd'):
            self._kernel_fft = self._template_fft.conjugate() * self._inv_psd
            self._kernel_normalization =  np.real(
                np.sum(self._kernel_fft * self._template_fft)
            ) * self._sampling_frequency / self._length

    def fit(self, trace):#The chi2 of this suppose to be 1/4 of original, since the integration is only on the real part
        trace_fft = sp_fft.rfft(trace) / self._sampling_frequency
        trace_filtered = self._kernel_fft * trace_fft
        amp0 = np.real(np.sum(trace_filtered)) * self._sampling_frequency / (self._length * self._kernel_normalization)

        chisq0 = np.real(np.vdot(trace_fft, trace_fft * self._inv_psd)) * self._sampling_frequency / self._length
        chisq = (chisq0 - amp0**2 * self._kernel_normalization) / (self._length - 2)

        return amp0, chisq

    def fit_with_shift(self, trace, allowed_shift_range=[-2000, 2000]):
        trace_fft = sp_fft.rfft(trace) / self._sampling_frequency
        trace_filtered = self._kernel_fft * trace_fft / self._kernel_normalization

        # A(t0) is the inverse FFT of the filtered signal
        trace_filtered_td = sp_fft.irfft(trace_filtered) * self._sampling_frequency

        # Compute chi^2_0 (independent of shift)
        chisq0 = np.real(np.vdot(trace_fft, trace_fft * self._inv_psd)) * self._sampling_frequency / self._length

        # Compute chi^2(t0) = chisq0 - A(t0)^2 * norm
        amp_series = trace_filtered_td*0.5#correct irfft and rfft
        chisq_series = chisq0 - amp_series**2 * self._kernel_normalization

        if allowed_shift_range is None:
            ind = np.arange(len(chisq_series))
        else:
            start = (self._length + allowed_shift_range[0]) % self._length
            stop = (allowed_shift_range[1] + 1) % self._length
            if start < stop:
                ind = np.arange(start, stop)
            else:
                ind = np.concatenate((np.arange(start, self._length), np.arange(0, stop)))

        best_ind = ind[np.argmin(chisq_series[ind])]
        amp = amp_series[best_ind]
        chisq = chisq_series[best_ind] / (self._length - 3)
        t0 = best_ind if best_ind < self._length // 2 else best_ind - self._length

        return amp, chisq, t0
    
