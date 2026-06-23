#Optimum filter V1.2, accelerated
import numpy as np
from scipy.fft import fft
from numba import njit
from numpy.fft import rfft, irfft, fft, ifft, fftfreq, rfftfreq

    
@njit(cache=True)
def _compute_amp(F, X, fs, N):
    # real(dot(F, X)) without building temporaries
    s = 0.0
    for k in range(N):
        s += F[k].real * X[k].real - F[k].imag * X[k].imag
    return s * fs / N

@njit(cache=True)
def _compute_chi0(X, S_unf, fs, N):
    # sum( |X|^2 / S_unf )
    s = 0.0
    for k in range(N):
        xr = X[k].real
        xi = X[k].imag
        s += (xr * xr + xi * xi) / S_unf[k]
    return s * fs / N
@njit(cache=True)
def _slide_and_eval(x, fs, F, S_unf, E, X, start, hop, steps, N, kernel_norm, amps, chisqs, out_offset):
    """
    Advances the sliding DFT 'steps' windows starting at 'start', in increments of 'hop',
    updating X in-place and writing amps/chisqs to output arrays beginning at out_offset.
    """
    t0 = start - hop  # matches the Python reference logic
    for s in range(steps):
        # perform 'hop' micro-steps
        for u in range(hop):
            t = t0 + u + s * hop
            # X = (X - x[t]/fs + x[t+N]/fs) * E
            a = x[t] / fs
            b = x[t + N] / fs
            for m in range(N):
                X[m] = (X[m] - a + b) * E[m]

        amp = _compute_amp(F, X, fs, N)
        chi0 = _compute_chi0(X, S_unf, fs, N)
        chisq = (chi0 - amp * amp * kernel_norm) / (N - 2)
        amps[out_offset + s] = amp
        chisqs[out_offset + s] = chisq

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
        
        
    def fit(self, trace):
        trace_fft = fft(trace, axis=-1)/self._sampling_frequency # V
        trace_filtered = self._filter_kernel * trace_fft
        amp = np.real(trace_filtered.sum(axis=-1)) * self._sampling_frequency / self._length
        chisq0 = np.real((trace_fft.conj() * trace_fft / self._noise_psd_unfolded).sum()) * self._sampling_frequency / self._length
        chisq = (chisq0 - amp**2 * self._kernel_normalization) / (self._length - 2) 

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

    
    def sliding_fit(self, trace_long, hop=1, reanchor_every=None):
        x = np.asarray(trace_long, dtype=np.float64)
        L = x.size
        N = int(self._length)
        fs = float(self._sampling_frequency)

        if N <= 0 or L < N:
            raise ValueError("Trace shorter than window length or invalid N.")
        if hop <= 0:
            raise ValueError("hop must be a positive integer.")

        # Number of windows
        num_windows = 1 + (L - N) // hop
        amps   = np.empty(num_windows, dtype=np.float64)
        chisqs = np.empty(num_windows, dtype=np.float64)

        # Twiddle for NumPy/SciPy FFT convention
        m = np.arange(N, dtype=np.float64)
        E = np.exp(2j * np.pi * m / N).astype(np.complex128)

        # Ensure stable, contiguous dtypes
        F     = np.asarray(self._filter_kernel, dtype=np.complex128)
        S_unf = np.asarray(self._noise_psd_unfolded, dtype=np.float64)
        kern_norm = float(self._kernel_normalization)

        # Initial N-point FFT (SciPy) — keep your 1/fs scaling
        X = fft(x[0:N]) / fs
        X = np.ascontiguousarray(X.astype(np.complex128))

        # First window outputs
        amp0  = (F.real @ X.real - F.imag @ X.imag) * fs / N
        chi00 = ((X.real * X.real + X.imag * X.imag) / S_unf).sum() * fs / N
        chisq0 = (chi00 - amp0 * amp0 * kern_norm) / (N - 2)
        amps[0] = amp0
        chisqs[0] = chisq0

        made = 1
        start = hop

        # Main loop: alternate between fast numba sliding and occasional SciPy re-anchors
        while start <= L - N:
            if reanchor_every and (made % reanchor_every == 0):
                # Recompute from scratch to limit numerical drift (SciPy FFT)
                X[:] = fft(x[start:start + N]) / fs
                amp = (F.real @ X.real - F.imag @ X.imag) * fs / N
                chi0 = ((X.real * X.real + X.imag * X.imag) / S_unf).sum() * fs / N
                chisq = (chi0 - amp * amp * kern_norm) / (N - 2)
                amps[made] = amp
                chisqs[made] = chisq
                made += 1
                start += hop
            else:
                # How many windows until the next reanchor or the end?
                if reanchor_every:
                    steps_to_reanchor = reanchor_every - (made % reanchor_every)
                else:
                    steps_to_reanchor = (L - N - start) // hop + 1
                steps_to_end = (L - N - start) // hop + 1
                steps = min(steps_to_reanchor, steps_to_end)
                if steps <= 0:
                    break

                _slide_and_eval(
                    x=x, fs=fs, F=F, S_unf=S_unf, E=E, X=X,
                    start=start, hop=hop, steps=steps, N=N, kernel_norm=kern_norm,
                    amps=amps, chisqs=chisqs, out_offset=made
                )
                made  += steps
                start += steps * hop

        return amps, chisqs