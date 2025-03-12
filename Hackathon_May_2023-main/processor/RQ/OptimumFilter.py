import numpy as np
from numpy.fft import rfft, irfft, fft, ifft, fftfreq, rfftfreq

class OptimumFilter():
    '''
    A class for performing Optimum Filter fits to traces.
    
    Supports fits with and withod template shift.
    
    To initialize one needs to specify pulse template, noise PSD and the sampling frequency of the traces  
    '''
    
    def __init__(self, template, noise_psd, sampling_frequency):
        '''
        Initialization of the OF class.
        Sets pulse template, noise PSD and sampling frequency to be used for the fits
        
        Parameters
        ----------
        template : 1d array
            Pulse template. Defines the length L of the traces
        noise_psd : 1d array of length L/2+1 if L is even and (L+1)/2 if L is odd
            PSD of noise, calculated as a real-valued FFT: np.mean(np.abs(np.fft.rfft(noise_traces))**2.0, axis=0)/(sampling_frequency*len(noise_traces[0]))
        sampling_frequency : float
            Frequency of the sampling in [Hz]

        Returns
        -------
        obj
            OptimumFilter object
        '''
        self._template = template
        self._noise_psd = noise_psd
        self._sampling_frequency = sampling_frequency
        self._update_state()
        
    def set_template(self, template):
        '''
        Changes the pulse template
        
        Parameters
        ----------
        
        template: 1d array
            Pulse template. Defines the length of the traces 
        '''
        self._template = template
        self._update_state()
        
    def set_noise_psd(self, noise_psd):
        '''
        Changes the noise PSD
        
        Parameters
        ----------
        noise_psd : 1d array of length L/2+1 if L is even and (L+1)/2 if L is odd (L is the length of the template)
            PSD of noise, calculated as a real-valued FFT: np.mean(np.abs(np.fft.rfft(noise_traces))**2.0, axis=0)/(sampling_frequency*len(noise_traces[0]))
        '''
        self._noise_psd = noise_psd
        self._update_state()
        
    def _update_state(self):
        # length of the template. All the traces must be of the same length
        self._length = len(self._template)
        
        # unfolding the NoisePSD from the real-valued case to the full FFT, including the negative frequencies
        if self._length%2==0:
            self._noise_psd_unfolded = np.concatenate(([np.inf],
                                                       self._noise_psd[1:-1]/2,
                                                       [self._noise_psd[-1]],
                                                       self._noise_psd[-2:0:-1]/2))
        else:
            self._noise_psd_unfolded = np.concatenate(([np.inf],
                                                       self._noise_psd[1:]/2,
                                                       self._noise_psd[-1:0:-1]/2))
            
        
        # applying full FFT to the template
        self._template_fft = fft(self._template)/self._sampling_frequency
        
        # Calculating the filter kernel and normaliation
        # the kernel is defined as Phi = S* / J, where S* is the conjugate of the template in frequency domain and J is the full noise PSD
        self._kernel_fft = self._template_fft.conjugate() / self._noise_psd_unfolded
        # the normalization is sum_k S_k S*_k / J_k (k - frequency component index)
        self._kernel_normalization = np.real(np.dot(self._kernel_fft, self._template_fft))*self._sampling_frequency/self._length 
        
    def fit(self, trace):
        '''
        Performs a fit without allowing a shift of the template.
        
        Parameters
        ----------
        trace : 1d array
            a trace to apply the fitting to
            
        Returns
        -------
        amp0 : float
            amplitude of the template
        chisq : float
            chi-squared divided by the number of degrees of freedom
        '''
        # TODO: maybe it is possible to vectorize this code, so it runs for for multiple traces simultaneously?
        
        # full FFT of the input trace, V
        trace_fft = fft(trace, axis=-1)/self._sampling_frequency
        # filtered trace in freq. domain: Phi V / Norm
        trace_filtered = self._kernel_fft * trace_fft / self._kernel_normalization
        
        # one part of the chi2 calculation. This is not the chi2 yet: V V* / J
        chisq0 = np.real(
            np.dot(trace_fft.conjugate()/self._noise_psd_unfolded,
                   trace_fft)*self._sampling_frequency/self._length
        )
        
        # Calculating the amplitude using an analtical solution of the chi2 minimization problem: A = Sum (Phi V / Norm)
        amp0 = np.real(np.sum(
            trace_filtered, axis=-1
        ))*self._sampling_frequency/self._length


        # final chisq
        # self._length-1 is the assumed number of degrees of freedom
        chisq = (chisq0 - (amp0**2)*self._kernel_normalization)/(self._length-1)
        
        return amp0, chisq
    
    def fit_with_shift(self, trace, allowed_shift_range=None):
        '''
        Performs a fit, allowing the template to shift left and right, rolling around the edges.
        Has two free parameters: the template amplitude a the time shift, measured in samples.
        Again, being more precise, this is not a shift, but an np.roll operation
        
        Parameters
        ----------
        trace : 1d array
            a trace to apply the fitting to
        
        allowed_shift_range : tuple of two floats
            The range of allowed templates shifts, in samples.
            E.g. with allowed_shift_range = (-10,10) the template is allowed to be rolled by 10 samples left and right
            
        Returns
        -------
        amp0 : float
            amplitude of the template
        chisq : float
            chi-squared divided by the number of degrees of freedom
        t0 : int
            the determined template time shift measured in time samples (not in seconds!)
        '''
        # TODO: maybe it is possible to vectorize this code, so it runs for for multiple traces simultaneously?
 
        # full FFT of the input trace, V
        trace_fft = fft(trace, axis=-1)/self._sampling_frequency # V
        
        # filtered trace in freq. domain: Phi V / Norm
        trace_filtered = self._kernel_fft * trace_fft / self._kernel_normalization # Phi * V
        
        # inverse FFT to get the filtered trace in time domain
        trace_filtered_td = np.real(
            ifft(trace_filtered, axis=-1)
        ) * self._sampling_frequency
        
        # first half of the chi2 calculation
        chi0 = np.real(
            np.dot(trace_fft.conjugate()/self._noise_psd_unfolded,
                   trace_fft)*self._sampling_frequency/self._length
        )
        
        # second part of the chi2 calculation 
        chit_withdelay = (trace_filtered_td**2) * self._kernel_normalization
        # an array of chisq values calculated with different assumed template shifts
        chi = chi0 - chit_withdelay
        
        
        # indeces of chisq values that are within the allowed range
        if allowed_shift_range is None:
            ind = np.arange(len(chi))
        else:
            ind = np.concatenate((np.arange(self._length+allowed_shift_range[0], self._length),
                                  np.arange(allowed_shift_range[1]+1)))
        
        # locating the smallest chisq in the allowed range
        best_ind = ind[np.argmin(chi[ind], axis=-1)]
        
        # the amplitude is equal to the filtered trace in time domain, at the location of best chi2 
        amp = trace_filtered_td[best_ind]
        # assume number of degrees of freedom is L-2
        chisq = chi[best_ind]/(self._length-2)
        # if the index is more than half of L, than it should be a negative shift
        t0 = best_ind if best_ind<self._length//2 else best_ind-self._length
        
        return amp, chisq, t0