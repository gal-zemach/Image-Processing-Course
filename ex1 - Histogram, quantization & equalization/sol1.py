import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray


GS_REP = 1
RGB_REP = 2
MAX_VALUE = 256

COLOR_CELL = 2
GS = 2

Y_CH = 0
I_CH = 1
Q_CH = 2

YIQ_MATRIX = np.array([[0.299, 0.584, 0.114], [0.596, -0.275, -0.321],
                       [0.212, -0.523, 0.311]])


def int2float(img_int):
    """
    Returns the float representation [0...1] of a given int image [0...255].
    """
    img = img_int.astype(np.float64)
    img /= (MAX_VALUE - 1)
    return img


def float2int(img_float):
    """
    Returns the int representation [0...255] of a given float image [0...1].
    """
    img = img_float * (MAX_VALUE - 1)
    img = img.astype(int)
    return img


def read_image(filename, representation):
    """
    Reads an image file in a given representation and returns it.
    """
    img = imread(filename)
    img = int2float(img)
    if representation == GS_REP:
        img = rgb2gray(img)
    return img


def imdisplay(filename, representation):
    """
    displays an image from a given filename in the given representation.
    """
    img = read_image(filename, representation)
    if representation == GS_REP:
        plt.imshow(img, cmap=plt.cm.gray)
    else:
        plt.imshow(img)


def multiply_by_left_matrix(matrix, img):
    """
    returns matrix * img, when matrix is a 3*3 matrix and
    img is 3d matrix containing a 3-channel image.
    """
    first = np.inner(matrix[0], img)
    second = np.inner(matrix[1], img)
    third = np.inner(matrix[2], img)

    res = np.dstack((first, second, third))
    return res


def rgb2yiq(im_rgb):
    """
    Returns a float RGB image's YIQ representation.
    """
    return multiply_by_left_matrix(YIQ_MATRIX, im_rgb)


def yiq2rgb(im_yiq):
    """
    Returns a float YIQ image's RGB representation.
    """
    return multiply_by_left_matrix(np.linalg.inv(YIQ_MATRIX), im_yiq)


def is_grayscale(img):
    """
    Returns true iff the image is a grayscale image.
    """
    return len(img.shape) == GS


def get_gray_channel(im_rgb):
    """
    Returns an image's gray channel
    """
    if not is_grayscale(im_rgb):
        im_yiq = rgb2yiq(im_rgb)
        return im_yiq[:, :, Y_CH].copy()
    return im_rgb.copy()


def update_gray_channel(im_rgb, new_gray):
    """
    Changes the given float image's gray channel to new_gray.
    """
    im_new = new_gray
    if not is_grayscale(im_rgb):
        im_yiq = rgb2yiq(im_rgb)
        new_im_yiq = np.dstack((new_gray, im_yiq[:, :, I_CH],
                                im_yiq[:, :, Q_CH]))
        im_new = yiq2rgb(new_im_yiq)

    return im_new


def linear_stretch(cum_hist):
    """
    Linearly Stretches a given cumulative histogram.
    """
    m = cum_hist.nonzero()[0][0]
    res = cum_hist - cum_hist[m]
    factor = (MAX_VALUE - 1) / (cum_hist[(MAX_VALUE - 1)] - cum_hist[m])
    res = np.multiply(res, factor)
    return res


def histogram_equalize(im_orig):
    """
    Equalizes an image's histogram (thus gray levels).
    Returns: The equalized image, the image's original histogram,
             the equalized image's histogram.
    """
    img = get_gray_channel(im_orig)
    img = float2int(img)
    
    # step1: computing histogram
    hist_orig, bins = np.histogram(img, bins=np.arange(MAX_VALUE + 1))

    # step2: computing cumulative histogram
    cum_hist = np.cumsum(hist_orig)
    
    # step3+4: Normalizing cumulative histogram and multiplying by
    #          the maximal gray level
    norm_factor = (MAX_VALUE - 1) / img.size
    cum_hist = np.multiply(cum_hist, norm_factor)
    
    # step5: Verifying values are in the right range
    if (int(np.amin(cum_hist)) != 0) or \
            (int(np.amax(cum_hist)) != MAX_VALUE - 1):
        cum_hist = linear_stretch(cum_hist)

    # step6: Round values
    cum_hist = np.round(cum_hist)

    # step7: Map image intensity values using histogram
    im_eq = cum_hist[img]

    hist_eq = np.histogram(im_eq, bins=np.arange(MAX_VALUE + 1))[0]
    im_eq = int2float(im_eq)
    im_eq = update_gray_channel(im_orig, im_eq)

    return im_eq, hist_orig, hist_eq


def guess_first_z(n_quant, hist):
    """
    Guesses n_quant + 1 z values based on a image's histogram.
    """
    cum_hist = np.cumsum(hist)
    factor = cum_hist[MAX_VALUE - 1] // n_quant
    z = [0]
    for i in range(1, n_quant):
        color_index = np.where(cum_hist > i * factor)
        idx = color_index[0][0]
        z.append(idx)
    z.append(MAX_VALUE - 1)
    return np.array(z)


def calculate_q(z, hist, hist_times_color):
    """
    Returns q values based on the given z values.
    hist: the image's histogram
    hist_times_color: the image's histogram where every bin is multiplied
                      by it's gray value
    """
    # calculate matrix for summing the histogram form z_i to z_{i + 1}
    sum_matrix = []
    for i in range(len(z) - 1):
        a = np.zeros(z[i])
        b = np.ones(z[i + 1] - z[i] + 1)
        c = np.zeros(z[len(z) - 1] - z[i + 1])
        vec = np.concatenate((a, b, c))
        if i:
            sum_matrix = np.vstack((sum_matrix, vec))
        else:
            sum_matrix = vec
    sum_matrix = np.transpose(sum_matrix)

    # calculate q values
    q_numer = np.dot(hist_times_color, sum_matrix)
    q_denom = np.dot(hist, sum_matrix)
    q = np.divide(q_numer, q_denom).astype(int)

    return q


def calculate_error(hist, z, q):
    """
    Calculates a given quantized image's error based on
    it's histogram and it's z & q values.
    """
    color_diff = np.arange(MAX_VALUE)
    for i in range(0, len(q)):
        color_diff[z[i]:z[i + 1]] -= q[i]
    color_diff[MAX_VALUE - 1] -= q[len(q) - 1]
    color_diff = np.square(color_diff)
    
    return np.sum(np.multiply(hist, color_diff))


def quantize(im_orig, n_quant, n_iter):
    """
    Quantizes a float image's colors to n_quant gray levels
    using n_iter iterations.
    Returns: The quantized image, an array with each iteration's error value.
    """
    img = get_gray_channel(im_orig)
    img = float2int(img)

    hist, bins = np.histogram(img, bins=np.arange(MAX_VALUE + 1))
    hist_times_color = hist * np.arange(MAX_VALUE)
    z = guess_first_z(n_quant, hist)
    error, q = [], []
    for i in range(0, n_iter):
        q = calculate_q(z, hist, hist_times_color)

        new_z = [0]
        for j in range(1, n_quant):
            new_z.append((q[j - 1] + q[j]) // 2)
        new_z.append(MAX_VALUE - 1)

        error.append(calculate_error(hist, new_z, q))

        if np.array_equal(z, new_z):
            break
        z = new_z

    lut = np.zeros(MAX_VALUE)
    for i in range(0, len(q)):
        lut[z[i]:z[i + 1]] = q[i]
    lut[MAX_VALUE - 1] = q[len(q) - 1]
    
    im_quant = lut[img.astype(int)]
    im_quant = int2float(im_quant)
    im_quant = update_gray_channel(im_orig, im_quant)
    
    return im_quant, error
