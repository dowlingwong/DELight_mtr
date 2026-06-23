import numpy as np
import scipy.fft as sp_fft

class JointChannelOF:

    def __init__(self, templates, noise_psds, sampling_frequency):
        """
        templates: list of 1D numpy arrays, one per channel
        noise_psds: list of 1D numpy arrays, one per channel
        """
        self._sampling_frequency = sampling_frequency
        self._n_channels = len(templates)
        self._length = len(templates[0])

        self.set_templates(templates)
        self.set_noise_psds(noise_psds)

    def set_templates(self, templates):
        self._templates = templates
        self._template_ffts = [sp_fft.rfft(tpl) / self._sampling_frequency for tpl in templates]
        self._update_kernels()

    def set_noise_psds(self, noise_psds):
        self._noise_psds = noise_psds

        self._inv_psds = []
        for psd in noise_psds:
            inv_psd = np.zeros_like(psd)
            inv_psd[1:] = 1.0 / (psd[1:] + 1e-30)
            if self._length % 2 == 0:
                inv_psd[-1] = 1.0 / (psd[-1] + 1e-30)
            inv_psd[0] = 0.0
            self._inv_psds.append(inv_psd)

        self._update_kernels()

    def _update_kernels(self):
        if hasattr(self, '_template_ffts') and hasattr(self, '_inv_psds'):
            self._kernels = []
            norms = []
            for tpl_fft, inv_psd in zip(self._template_ffts, self._inv_psds):
                kernel = tpl_fft.conjugate() * inv_psd
                norm = np.real(np.sum(kernel * tpl_fft)) * self._sampling_frequency / self._length
                self._kernels.append(kernel)
                norms.append(norm)

            self._kernel_normalization = np.sum(norms)

    def fit_with_shift(self, traces, allowed_shift_range=[0, 0]):
        """
        traces: list of 1D numpy arrays, one per channel
        """
        filtered_traces_fft = []
        trace_ffts = []
        chisq0_total = 0.0

        for trace, kernel, norm, inv_psd in zip(
            traces, self._kernels,
            [np.real(np.sum(k * t)) * self._sampling_frequency / self._length for k, t in zip(self._kernels, self._template_ffts)],
            self._inv_psds
        ):
            tr_fft = sp_fft.rfft(trace) / self._sampling_frequency
            trace_ffts.append(tr_fft)
            filtered = kernel * tr_fft / norm
            filtered_traces_fft.append(filtered)

            chisq0 = np.real(np.vdot(tr_fft, tr_fft * inv_psd)) * self._sampling_frequency / self._length
            chisq0_total += chisq0

        # Sum of inverse FFTs of all channels => joint A(t0)
        amp_series = 0.5 * sum(sp_fft.irfft(filt) * self._sampling_frequency for filt in filtered_traces_fft)

        chisq_series = chisq0_total - amp_series**2 * self._kernel_normalization

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
