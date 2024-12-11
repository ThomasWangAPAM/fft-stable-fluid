# fftfreq()
# fft2D()
# ifft2D()


import numpy

def freq(n: int, d: float = 1.0) -> numpy.array:
    """Compute the Discrete Fourier Transform frequencies.

    Args:
        n (int): 
            The size of the FFT, which determines the number of frequency bins.
            Must be a positive integer. This corresponds to the number of samples 
            in the input data that will be used in FFT computation.
        d (float, optional): 
            The sampling interval between consecutive points in the input data.
            Default to 1.0, which corresponds to frequecies of cycles per sample.
            Change to other values when there is temporal meaning in data so that
            frequency can be interpreted appropriately (e.g. Hz).
            
    Returns:
        numpy.array: 
            An array of frequency bins of length `n` that includes both positive and 
            negative frequencies. 
    
    Notes:
        - The first element corresponds to the DC component (0 Hz or 0 cycles/unit).
        - The positive frequencies are arranged in increasing order from index 0 to 
          `n//2`.
        - The negative frequencies are arranged in decreasing order from `-n//2` 
          to -1, following the Nyquist frequency if `n` is even.
    """
    coeff = 1.0 / (n * d)
    pos = numpy.arange(0, (n - 1) // 2 + 1, dtype=int)
    neg = numpy.arange(-(n // 2), 0, dtype=int)
    
    return numpy.concatenate((pos, neg)) * coeff


def dft():
    pass

def fft():
    pass

def ifft():
    pass 

def fft2():
    pass 


def ifft2():
    pass 