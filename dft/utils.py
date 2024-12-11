# dft/utils.py
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


def _dft(input: numpy.ndarray, axis: int = 0, forward: bool = True) -> numpy.ndarray:
    """
    Compute the Discrete Fourier Transform (DFT) or its inverse along a specified axis.

    This is a helper function that calculates the DFT or inverse DFT based on the `forward` flag.
    It uses the matrix multiplication approach to perform the transform.

    Parameters:
        input (numpy.ndarray): The input array to be transformed. Can be multi-dimensional.
        axis (int): The axis along which to compute the DFT. Default is 0.
        forward (bool): If True, computes the forward DFT. If False, computes the inverse DFT.
                        Default is True.

    Returns:
        numpy.ndarray: The transformed array with the same shape as the input.

    """
    # Move the specified axis to the second-to-last axis and expand a singleton dimension at the end
    # This is necessary because the @ operator aligns the last axis of the first matrix
    # with the second-to-last axis of the second matrix during matrix multiplication.
    f = numpy.moveaxis(input, axis, -1)[..., None]  

    N = f.shape[-2]

    # Construct the DFT matrix
    # For forward transform: exp(-2j * pi * k * n / N)
    # For inverse transform: exp(+2j * pi * k * n / N)
    DFT = numpy.exp(
        (-1)**forward * 2j * numpy.pi * numpy.arange(N) * numpy.arange(N)[..., None] / N
    )

    # Perform the matrix multiplication (DFT @ f) and remove the singleton dimension
    # Restore the original axis order after the computation
    result = numpy.moveaxis((DFT @ f)[..., -1], -1, axis)

    # Normalize by N for the inverse transform
    return result if forward else result / N


def dft(input: numpy.ndarray, axis: int = 0) -> numpy.ndarray:
    """
    Compute the Discrete Fourier Transform (DFT) along a specified axis, which
    converts data from its original domain to the frequency domain.

    Parameters:
        input (numpy.ndarray): The input array to be transformed. Can be multi-dimensional.
        axis (int): The axis along which to compute the DFT. Default is 0.

    Returns:
        numpy.ndarray: The transformed array with the same shape as the input.
    """
    return _dft(input, axis, True)


def idft(input: numpy.ndarray, axis: int = 0) -> numpy.ndarray:
    """
    Compute the Inverse Discrete Fourier Transform (IDFT) along a specified axis, which
    converts data from frequency domain to the origal domain.

    Parameters:
        input (numpy.ndarray): The input array to be transformed. Can be multi-dimensional.
        axis (int): The axis along which to compute the IDFT. Default is 0.

    Returns:
        numpy.ndarray: The transformed array with the same shape as the input.
    """
    return _dft(input, axis, False)


def _fft():
    pass


def fft():
    pass


def ifft():
    pass 


def fft2():
    pass 


def ifft2():
    pass 