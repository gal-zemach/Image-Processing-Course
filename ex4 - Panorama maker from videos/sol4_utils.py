
import numpy as np
from scipy.misc import imread as imread
from scipy.ndimage import convolve
from skimage.color import rgb2gray
from scipy.signal import convolve2d


GS_REP = 1
RGB_REP = 2
MAX_VALUE = 256
BASE_KERNEL = np.array([1, 1])
SIZE_F = 2
MIN_DIM = 16


def read_image(filename, representation):
    """
    Reads an image file in a given representation and returns it.
    """
    im = imread(filename)
    if representation == GS_REP:
        im = rgb2gray(im)
    im = np.divide(im, MAX_VALUE - 1)
    return im


def _binoms(kernel_size):
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


def _gaussian_kernel_1d(kernel_size):
    """
    Returns a gaussian kernel of a given size.
    :param kernel_size: an odd integer.
    """
    kernel = _binoms(kernel_size)
    return np.divide(kernel, np.sum(kernel))


def _gaussian_kernel(kernel_size):
    """
    Returns a gaussian kernel of a given size.
    :param kernel_size: an odd integer.
    """
    curr_kernel = _binoms(kernel_size)
    curr_kernel = curr_kernel.reshape(kernel_size, 1)
    kernel2d = convolve2d(curr_kernel.transpose(), curr_kernel)
    kernel2d = np.divide(kernel2d, np.sum(kernel2d))
    return kernel2d


def blur_spatial(im, kernel_size):
    """
    Blurs an image using convolution with a gaussian kernel of a given size.
    :param kernel_size: an odd integer.
    """
    kernel = _gaussian_kernel(kernel_size)
    return convolve2d(im, kernel, mode="same", boundary="symm")


def _reduce_rows(im, filter_vec):
    """
    Reduces an image's rows to half.
    :param im: The image to downsize
    :param filter_vec: a filter to be used for blurring the image before down-sampling.
    :return: The downsized image (of size n/2 * n)
    """
    im_small = convolve(im, filter_vec)
    return im_small[::SIZE_F]


def _reduce(im, filter_vec):
    """
    Reduces an image's size to half.
    :param im: The image to downsize
    :param filter_vec: a filter to be used for blurring the image before down-sampling.
    :return: The downsized image
    """
    im_small = _reduce_rows(im, filter_vec)
    im_small = _reduce_rows(im_small.transpose(), filter_vec).transpose()
    return im_small


def _expand_rows(im, filter_vec):
    """
    Expands the image to twice the number of rows.
    :param im: The image to expand
    :param filter_vec: a filter to be used for blurring the image after expanding (2 * filter_vec is used).
    :return: The expanded image of size 2n * n
    """
    n, m = im.shape
    im_large = np.zeros((n * SIZE_F, m))
    im_large[::SIZE_F] = im
    im_large = convolve(im_large, SIZE_F * filter_vec)
    return im_large


def _expand(im, filter_vec):
    """
    Expands the image to twice it's size.
    :param im: The image to expand
    :param filter_vec: a filter to be used for blurring the image after expanding (2 * filter_vec is used).
    :return: The expanded image of size
    """
    im_large = _expand_rows(im, filter_vec)
    im_large = _expand_rows(im_large.transpose(), filter_vec).transpose()
    return im_large


def _image_is_large_enough(im):
    """
    Returns true iff both of the image's dimensions are bigger than MIN_DIM
    """
    return (im.shape[0] >= MIN_DIM) and (im.shape[1] >= MIN_DIM)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    (3.1)
    Builds a gaussian pyramid of an image.
    :param im: The image
    :param max_levels: The maximum number of levels in the pyramid.
    :param filter_size: The size of the gaussian filter to be used in the image reductions.
    :return: A python list containing the pyramid, The gaussian filter used.
    """
    filter_vec = _gaussian_kernel_1d(filter_size).reshape(filter_size, 1)
    pyr = [im]
    im_curr = _reduce(im, filter_vec)
    while _image_is_large_enough(im_curr) and len(pyr) < max_levels:
        pyr.append(im_curr)
        im_curr = _reduce(im_curr, filter_vec)

    return pyr, filter_vec.reshape(1, filter_size)


def _build_laplacian_pyramid(im, max_levels, filter_size):
    """
    (3.1)
    Builds a laplacian pyramid of an image.
    :param im: The image
    :param max_levels: The maximum number of levels in the pyramid.
    :param filter_size: The size of the gaussian filter to be used in the image reductions.
    :return: A python list containing the pyramid, The gaussian filter used.
    """
    filter_vec = _gaussian_kernel_1d(filter_size).reshape(filter_size, 1)
    pyr = []
    im_curr = im
    im_next = _reduce(im, filter_vec)
    while _image_is_large_enough(im_next) and len(pyr) < max_levels - 1:
        pyr.append(im_curr - _expand(im_next, filter_vec))
        im_curr = im_next
        im_next = _reduce(im_curr, filter_vec)

    pyr.append(im_curr)

    return pyr, filter_vec.reshape(1, filter_size)


def _laplacian_to_image(lpyr, filter_vec, coeff):
    """
    (3.2)
    Reconstructs an image from it's laplacian pyramid.
    :param lpyr: a laplacian pyramid
    :param filter_vec: The filter used in the pyramid's creation
    :param coeff: The coefficients to multiply each level of the pyramid in the building process.
    :return: The built image.
    """
    im = lpyr[-1]
    filter_vec = filter_vec.reshape(filter_vec.size, 1)
    for i in reversed(range(len(lpyr) - 1)):
        im = _expand(im, filter_vec) + coeff[i] * lpyr[i]

    return im


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    (4.0)
    Blend two given images using pyramids.
    :param im1: 1st image
    :param im2: 2nd image
    :param mask: A binary mask used for the blending
    :param max_levels: The number of levels to be used in creating the pyramids
    :param filter_size_im: size of the filter to be used when creating the images' pyramids
    :param filter_size_mask: size of the filter to be used when creating the mask's pyramid
    :return: The blended image
    """

    l_1, filter_vec = np.array(_build_laplacian_pyramid(im1, max_levels, filter_size_im))  # also taking filter_vec
    l_2 = np.array(_build_laplacian_pyramid(im2, max_levels, filter_size_im)[0])

    g_m = np.array(build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)[0])

    l_out = np.multiply(g_m, l_1) + np.multiply(1 - g_m, l_2)

    coeff = np.ones(max_levels)
    im_blend = _laplacian_to_image(l_out, filter_vec, coeff)

    return np.clip(im_blend, 0, 1)
