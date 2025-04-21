from numpy.fft import rfft, fft, ifft, fftfreq, rfftfreq
import numpy as np

def calculate_psd(traces, sampling_frequency=1.0):
    """Return the PSD of an n-dimensional array, assuming that we want the PSD of the last axis.
    Originally taken from https://github.com/spice-herald/QETpy/blob/master/qetpy/core/_noise.py
    
    Parameters
    ----------
    traces : array_like
        Array to calculate PSD of.
    sampling_frequency : float, optional
        Sample rate of the data being taken, assumed to be in units of Hz.
            
    Returns
    -------
    f : ndarray
        Array of sample frequencies
    psd : ndarray
        Power spectral density of traces. If traces are in units of A, then the PSD is in units of A^2/Hz.
        One can plot the psd with
            plt.loglog(f,psd)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD (A$^2$/Hz)')
            plt.show()
        
    """
    
    # calculate normalization for correct units
    norm = sampling_frequency * traces.shape[-1]

    # if folded_over = True, we calculate the Fourier Transform for only the positive frequencies
    if len(traces.shape)==1:
        psd = (np.abs(rfft(traces))**2.0)/norm
    else:
        psd = np.mean(np.abs(rfft(traces))**2.0, axis=0)/norm

    # multiply the necessary frequencies by two (zeroth frequency should be the same, as
    # should the last frequency when x.shape[-1] is odd)
    psd[1:traces.shape[-1]//2+1 - (traces.shape[-1]+1)%2] *= 2.0
    f = rfftfreq(traces.shape[-1], d=1.0/sampling_frequency)

    return f, psd