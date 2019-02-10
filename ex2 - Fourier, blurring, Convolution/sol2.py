import numpy as np
from scipy.misc import imread as imread
from scipy.signal import convolve2d
from skimage.color import rgb2gray


GS_REP = 1
MAX_VALUE = 256
FOURIER_DER_CONST = 2j * np.pi
BASE_KERNEL = np.array([1, 1])


def read_image(filename, representation):
    """
    Reads an image file in a given representation and returns it.
    """
    im = imread(filename)
    if representation == GS_REP:
        im = rgb2gray(im)
    return im


def dft_base(signal, inverse=False):
    """
    Performs column-wise 1D DFT (or IDFT) of a given signal
    :param signal: a 2D matrix
    :param inverse: if True will perform inverse DFT
    :return: a matrix of the same size with the column-wise transform
    """
    n, m = signal.shape
    x = np.arange(n)
    u = np.meshgrid(x, x)[1]
    ux = u * x
    if inverse:
        ux = np.exp(FOURIER_DER_CONST * ux / n)
        return np.einsum('ij,ik->kj', signal[:, :m], ux) / n

    ux = np.exp(-FOURIER_DER_CONST * ux / n)
    return np.einsum('ij,ik->kj', signal[:, :m], ux)


def DFT(signal):
    """
    Performs column-wise 1D DFT of a given 2D signal
    """
    return dft_base(signal)


def IDFT(fourier_signal):
    """
    Performs column-wise 1D inverse DFT of a given 2D signal
    """
    return dft_base(fourier_signal, True)


def DFT2(image):
    """
    Performs a 2D DFT of a given image.
    """
    res = DFT(image).transpose()
    return DFT(res).transpose()


def IDFT2(fourier_image):
    """
    Performs a 2D inverse DFT of a given image.
    """
    res = IDFT(fourier_image).transpose()
    return IDFT(res).transpose()


def magnitude(dx, dy):
    """
    Returns the magnitude of given derivatives in x & y
    """
    return np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2)


def conv_der(im):
    """
    Returns the magnitude of an image's derivative using convolution.
    """
    kernel = np.array([1, 0, -1]).reshape(3, 1)
    dx = convolve2d(im, kernel, mode="same", boundary="symm")
    dy = convolve2d(im, kernel.transpose(), mode="same", boundary="symm")
    return magnitude(dx, dy)


def fourier_dx(f_im):
    """
    Returns a fourier image's derivative in x direction.
    """
    n = f_im.shape[0]
    idx = np.arange(n) - n // 2
    idx = np.fft.ifftshift(idx)

    dx = idx.reshape(n, 1) * f_im
    dx *= FOURIER_DER_CONST / n

    return IDFT2(dx)


def fourier_der(im):
    """
    Returns the magnitude of an image's derivative using DFT.
    """
    f_im = DFT2(im)
    dx = fourier_dx(f_im)
    dy = fourier_dx(f_im.transpose()).transpose()

    return magnitude(dx, dy)


def binoms(kernel_size):
    """
    Returns the binomial coefficients of a given order.
    :param kernel_size: The order number.
    """
    if kernel_size > 1:
        curr_kernel = BASE_KERNEL
        for i in range(2, kernel_size):
            curr_kernel = np.convolve(curr_kernel, BASE_KERNEL)
        return curr_kernel
    return np.array([1])


def gaussian_kernel(kernel_size):
    """
    Returns a gaussian kernel of a given size.
    :param kernel_size: an odd integer.
    """
    curr_kernel = binoms(kernel_size)
    curr_kernel = curr_kernel.reshape(kernel_size, 1)
    kernel2d = convolve2d(curr_kernel.transpose(), curr_kernel)
    kernel2d = np.divide(kernel2d, np.sum(kernel2d))
    return kernel2d


def blur_spatial(im, kernel_size):
    """
    Blurs an image using convolution with a gaussian kernel of a given size.
    :param kernel_size: an odd integer.
    """
    kernel = gaussian_kernel(kernel_size)
    return convolve2d(im, kernel, mode="same", boundary="symm")


def blur_fourier(im, kernel_size):
    """
    Blurs an image using DFT with a gaussian kernel of a given size.
    :param kernel_size: an odd integer.
    """
    padded_kernel = np.zeros(im.shape)
    small_kernel = gaussian_kernel(kernel_size)

    # building gaussian kernel in the middle of an image sized matrix
    n, m = im.shape
    bg_center = (n // 2 + 1, m // 2 + 1)
    padded_kernel[bg_center[0] - 1, bg_center[1] - 1] = 1
    padded_kernel = convolve2d(padded_kernel, small_kernel, mode="same")
    f_kernel = DFT2(np.fft.ifftshift(padded_kernel))

    # calculating blurred image
    f_im = DFT2(im)
    blurred_f_im = np.multiply(f_im, f_kernel)
    return np.real(IDFT2(blurred_f_im))
