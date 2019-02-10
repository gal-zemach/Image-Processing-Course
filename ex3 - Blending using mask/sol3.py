
import numpy as np
from scipy.misc import imread as imread
from scipy.ndimage import convolve
from skimage.color import rgb2gray
from os.path import join, dirname
import matplotlib.pyplot as plt


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


def gaussian_kernel_1d(kernel_size):
    """
    Returns a gaussian kernel of a given size.
    :param kernel_size: an odd integer.
    """
    kernel = binoms(kernel_size)
    return np.divide(kernel, np.sum(kernel))


def reduce_rows(im, filter_vec):
    """
    Reduces an image's rows to half.
    :param im: The image to downsize
    :param filter_vec: a filter to be used for blurring the image before down-sampling.
    :return: The downsized image (of size n/2 * n)
    """
    im_small = convolve(im, filter_vec)
    return im_small[::SIZE_F]


def reduce(im, filter_vec):
    """
    Reduces an image's size to half.
    :param im: The image to downsize
    :param filter_vec: a filter to be used for blurring the image before down-sampling.
    :return: The downsized image
    """
    im_small = reduce_rows(im, filter_vec)
    im_small = reduce_rows(im_small.transpose(), filter_vec).transpose()
    return im_small


def expand_rows(im, filter_vec):
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


def expand(im, filter_vec):
    """
    Expands the image to twice it's size.
    :param im: The image to expand
    :param filter_vec: a filter to be used for blurring the image after expanding (2 * filter_vec is used).
    :return: The expanded image of size
    """
    im_large = expand_rows(im, filter_vec)
    im_large = expand_rows(im_large.transpose(), filter_vec).transpose()
    return im_large


def image_is_large_enough(im):
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
    filter_vec = gaussian_kernel_1d(filter_size).reshape(filter_size, 1)
    pyr = [im]
    im_curr = reduce(im, filter_vec)
    while image_is_large_enough(im_curr) and len(pyr) < max_levels:
        pyr.append(im_curr)
        im_curr = reduce(im_curr, filter_vec)

    return pyr, filter_vec.reshape(1, filter_size)


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    (3.1)
    Builds a laplacian pyramid of an image.
    :param im: The image
    :param max_levels: The maximum number of levels in the pyramid.
    :param filter_size: The size of the gaussian filter to be used in the image reductions.
    :return: A python list containing the pyramid, The gaussian filter used.
    """
    filter_vec = gaussian_kernel_1d(filter_size).reshape(filter_size, 1)
    pyr = []
    im_curr = im
    im_next = reduce(im, filter_vec)
    while image_is_large_enough(im_next) and len(pyr) < max_levels - 1:
        pyr.append(im_curr - expand(im_next, filter_vec))
        im_curr = im_next
        im_next = reduce(im_curr, filter_vec)

    pyr.append(im_curr)

    return pyr, filter_vec.reshape(1, filter_size)


def laplacian_to_image(lpyr, filter_vec, coeff):
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
        im = expand(im, filter_vec) + coeff[i] * lpyr[i]

    return im


def stretch_values(im):
    """
    Stretches an image's values to [0,1].
    :param im: an image in float64
    :return: The stretched image
    """
    min_val, max_val = np.amin(im), np.amax(im)
    res = im - min_val
    if max_val - min_val:
        np.divide(res, max_val - min_val)
    return res


def render_pyramid(pyr, levels):
    """
    (3.3)
    Returns an image containing a number of levels of a pyramid.
    :param pyr: a gaussian or laplacian pyramid
    :param levels: The number of levels to use
    :return: An image containing the requested pyramid levels.
    """
    res = stretch_values(pyr[0])
    for i in range(1, min(levels, len(pyr))):
        padded = np.pad(stretch_values(pyr[i]), ((0, res.shape[0] - pyr[i].shape[0]), (0, 0)), mode="constant")
        res = np.concatenate((res, padded), axis=1)
    return res


def display_pyramid(pyr, levels):
    """
    (3.3)
    Displays an image containing a number of levels of a pyramid.
    :param pyr: a gaussian or laplacian pyramid
    :param levels: The number of levels to display
    """
    to_display = render_pyramid(pyr, levels)
    plt.imshow(to_display, cmap="gray")
    plt.show()


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

    l_1, filter_vec = np.array(build_laplacian_pyramid(im1, max_levels, filter_size_im))  # also taking filter_vec
    l_2 = np.array(build_laplacian_pyramid(im2, max_levels, filter_size_im)[0])

    g_m = np.array(build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)[0])

    l_out = np.multiply(g_m, l_1) + np.multiply(1 - g_m, l_2)

    coeff = np.ones(max_levels)
    im_blend = laplacian_to_image(l_out, filter_vec, coeff)

    return np.clip(im_blend, 0, 1)


def relpath(filename):
    """
    Returns the full path for a given filename with relative path.
    """
    return join(dirname(__file__), filename)


def blending_example_base(filename1, filename2, filename3):
    """
    (4.1)
    Used in the blending examples. Blends two RGB images using a given mask.
    :param filename1: filename for the 1st rgb image to blend.
    :param filename2: filename for the 2nd rgb image to blend.
    :param filename3: filename for the mask to be used in the blending.
    :return: The 3 above images and the blended image.
    """
    max_levels = 8
    filter_size = 21
    im1 = read_image(relpath(filename1), RGB_REP)
    im2 = read_image(relpath(filename2), RGB_REP)
    mask = read_image(relpath(filename3), GS_REP).astype(np.bool)

    im_blend = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, max_levels, filter_size, filter_size)
    for i in range(1, 3):
        curr_blend = pyramid_blending(im1[:, :, i], im2[:, :, i], mask, max_levels, filter_size, filter_size)
        im_blend = np.dstack((im_blend, curr_blend))

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.imshow(im_blend)
    plt.show()

    return im1, im2, mask, im_blend


def blending_example1():
    """
    (4.1)
    Performs an example blending of 2 images and shows it.
    Returns the images, the used mask and the results.
    """
    filename1 = "externals/01_image1.png"
    filename2 = "externals/01_image2.png"
    filename3 = "externals/01_mask.png"
    return blending_example_base(filename1, filename2, filename3)


def blending_example2():
    """
    (4.1)
    Performs another example of blending of 2 images and shows it.
    Returns the images, the used mask and the results.
    """
    filename1 = "externals/02_image1.png"
    filename2 = "externals/02_image2.png"
    filename3 = "externals/02_mask.png"
    return blending_example_base(filename1, filename2, filename3)
