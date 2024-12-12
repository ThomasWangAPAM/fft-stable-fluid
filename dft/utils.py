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


def _dft(input: numpy.ndarray, axis: int, forward: bool) -> numpy.ndarray:
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


# pre define DFT matrix for length of 16
DFT_16 = numpy.exp(-2j * numpy.pi * numpy.arange(16) * numpy.arange(16)[:, None] / 16)
TWIDDLE_FACTOR={N: numpy.exp(-2j * numpy.pi * numpy.arange(N/2)/ N) for N in numpy.array([1024//32,1024//16,1024//8,1024//4,1024//2,
                                                                  1024,
                                                                  1024*2,1024*4,1024*8,1024*16,1024*32,1024*64,1024*128])}

def dft(input: numpy.ndarray, axis: int = -1) -> numpy.ndarray:
    """
    Compute the Discrete Fourier Transform (DFT) along a specified axis, which
    converts data from its original domain to the frequency domain.

    Parameters:
        input (numpy.ndarray): The input array to be transformed. Can be multi-dimensional.
        axis (int): The axis along which to compute the DFT. Default is -1.

    Returns:
        numpy.ndarray: The transformed array with the same shape as the input.
    """
    return _dft(input, axis, True)


def idft(input: numpy.ndarray, axis: int = -1) -> numpy.ndarray:
    """
    Compute the Inverse Discrete Fourier Transform (IDFT) along a specified axis, which
    converts data from frequency domain to the origal domain.

    Parameters:
        input (numpy.ndarray): The input array to be transformed. Can be multi-dimensional.
        axis (int): The axis along which to compute the IDFT. Default is -1.

    Returns:
        numpy.ndarray: The transformed array with the same shape as the input.
    """
    return _dft(input, axis, False)


def _fft(input: numpy.ndarray, axis: int, forward: bool) -> numpy.ndarray:
    """
    Compute the Fast Fourier Transform (FFT) using the Cooley-Tukey algorithm.

    This function implements an iterative version of the Cooley-Tukey FFT algorithm,
    which is efficient for input sizes that are powers of 2.

    Parameters:
    -----------
    input (numpy.ndarray): The input array to be transformed. Can be multi-dimensional.
    axis (int): The axis along which to compute the FFT. Default is -1.
    forward (bool): If True, computes the forward FFT. If False, computes the inverse DFT.
                    Default is True.

    Returns:
    --------
    Returns:
        numpy.ndarray: The transformed array with the same shape as the input.

    Raises:
    -------
    ValueError
        If the size of the input along the specified axis is not a power of 2.

    Notes:
    ------
        - This implementation uses a base case of N_min = 16, below which it switches
          to a direct DFT computation.
        - The algorithm uses precomputed twiddle factors for efficiency.
        - For inverse FFT (forward=False), the output is scaled by 1/N.
    """

    N_min = 16
    N = input.shape[axis]
    
    if N & (N - 1) != 0:
        raise ValueError("size of input must be a power of 2 for original Cooley–Tukey")
    
    if N <= N_min:
        return _dft(input, axis, forward)

    f = numpy.moveaxis(input, axis, -1)
    
    DFT16 = DFT_16 if forward else numpy.conjugate(DFT_16)
    F = DFT16 @ f.reshape((*f.shape[:-1], N_min, -1))
    
    while F.shape[-2] < N:
        F_even = F[..., :F.shape[-1] // 2]
        F_odd = F[..., F.shape[-1] // 2:]
        factor = TWIDDLE_FACTOR[2 * F.shape[-2]][..., None]
        factor_ = factor if forward else numpy.conjugate(factor)
        t = factor_ * F_odd
        F = numpy.concatenate((F_even+t, F_even-t), axis=-2)
    
    result = numpy.moveaxis(F[..., -1], -1, axis)
    return result if forward else result / N


def fft(input: numpy.ndarray, axis: int = -1) -> numpy.ndarray:
    """
    Compute the Fast Fourier Transform (FFT) along a specified axis, which
    converts data from its original domain to the frequency domain.

    Parameters:
        input (numpy.ndarray): The input array to be transformed. Can be multi-dimensional.
        axis (int): The axis along which to compute the FFT. Default is -1.

    Returns:
        numpy.ndarray: The transformed array with the same shape as the input.
    """
    return _fft(input, axis, True)


def ifft(input: numpy.ndarray, axis: int = -1) -> numpy.ndarray:
    """
    Compute the Inverse Fast Fourier Transform (IFFT) along a specified axis, which
    converts data from frequency domain to the origal domain.

    Parameters:
        input (numpy.ndarray): The input array to be transformed. Can be multi-dimensional.
        axis (int): The axis along which to compute the IFFT. Default is -1.

    Returns:
        numpy.ndarray: The transformed array with the same shape as the input.
    """
    return _fft(input, axis, False)


def fft2(input: numpy.ndarray, axes: tuple[int] = (-1, -2)):
    """
    Compute the Fast Fourier Transform (FFT) along two specified axes, which
    converts data from its original domain to the frequency domain.

    Parameters:
        input (numpy.ndarray): The input array to be transformed. Can be multi-dimensional.
        axis (tuple[int]): The axes along which to compute the 2D-FFT. Default is (-1, -2).

    Returns:
        numpy.ndarray: The transformed array with the same shape as the input.
    """
    for a in axes:
        input = _fft(input, a, True)
    return input


def ifft2(input: numpy.ndarray, axes: tuple[int] = (-1, -2)):
    """
    Compute the Inverse Fast Fourier Transform (IFFT) along two specified axes, which
    converts data from frequency domain to the origal domain.

    Parameters:
        input (numpy.ndarray): The input array to be transformed. Can be multi-dimensional.
        axis (tuple[int]): The axes along which to compute the 2D-IFFT. Default is (-1, -2).

    Returns:
        numpy.ndarray: The transformed array with the same shape as the input.
    """
    for a in axes:
        input = _fft(input, a, False)
    return input 