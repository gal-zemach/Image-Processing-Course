
import numpy as np
import sol5_utils

from scipy.misc import imread as imread
from scipy.ndimage.filters import convolve
from skimage.color import rgb2gray
from skimage import img_as_float

from keras.layers import Input, merge, Convolution2D, Activation
from keras.models import Model
from keras.optimizers import Adam


# read_image
GS_REP = 1
RGB_REP = 2
MAX_VALUE = 256

# load_dataset
SUB_VALUE = 0.5

# res_block
CONV_KERNEL_SIZE = 3

# build_nn_model
CHANNELS_GS = 1

# train_model
TRAINING_SET_SPLIT = 0.8
BETA2_VAL = 0.9

# train_model_quick_mode
QUICK_BATCH_SIZE = 10
QUICK_SAMPLES_PER_EPOCH = 30
QUICK_EPOCHS = 2
QUICK_VALID_SAMPLES = 30

# learn_denoising_model
LDN_PATCH_SIZE = 24
LDN_CHANNELS = 48
LDN_MIN_SIG = 0
LDN_MAX_SIG = 0.2
LDN_EPOCHS = 5

# learn_deblurring_model
LDB_PATCH_SIZE = 16
LDB_CHANNELS = 32
LDB_KERNEL_SIZE = 7
LDB_EPOCHS = 10

# used by both denoising & deblurring networks
BATCH_SIZE = 100
SAMPLES_PER_EPOCH = 10000
VALID_SAMPLES = 1000


def read_image(filename, representation):
    """
    Reads an image file in a given representation and returns it.
    """
    im = imread(filename)
    if representation == GS_REP:
        im = rgb2gray(im)
    im = img_as_float(im)
    return im


# 3
def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    Returns a generator that yields batches of source (corrupted) images and a their target (uncorrupted) counterparts.
    :param filenames: A list of filenames
    :param batch_size: The number of images in each batch
    :param corruption_func: A function for randomly corrupting the images
    :param crop_size: A tuple (n,m) indicating the size of image slices to return.
    :return: Two lists of shape (batch_size, 1, n, m) with the source and target image slices.
    """
    rows, cols = crop_size
    im_dict = {}

    while True:

        target_batch = []
        source_batch = []

        for i in range(batch_size):

            idx = np.random.randint(len(filenames))
            if filenames[idx] not in im_dict:
                im_dict[filenames[idx]] = read_image(filenames[idx], GS_REP)

            im = im_dict[filenames[idx]]
            row_0, col_0 = np.random.randint(im.shape[0] - rows), np.random.randint(im.shape[1] - cols)
            im_slice = im[row_0:row_0 + rows, col_0:col_0 + cols]

            target_batch.append((im_slice - SUB_VALUE)[np.newaxis, ...])
            source_batch.append((corruption_func(im_slice) - SUB_VALUE)[np.newaxis, ...])

        yield np.array(source_batch), np.array(target_batch)


# 4
def resblock(input_tensor, num_channels):
    """
    Represents one residual block in the ResNet.
    :param input_tensor: symbolic tensor
    :param num_channels: number of channels for the convolution layers
    :return: symbolic output tensor
    """
    a = Convolution2D(num_channels, CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, border_mode='same')(input_tensor)
    b = Activation('relu')(a)
    c = Convolution2D(num_channels, CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, border_mode='same')(b)
    output_tensor = merge([c, input_tensor], mode='sum')
    return output_tensor


# 4
def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    Returns an untrained Keras model of a ResNet network.
    :param height: input images' height
    :param width: input images' width
    :param num_channels: number of channels for the convolution layers
    :param num_res_blocks: number of residual blocks in the network
    :return: an untrained Keras model
    """
    a = Input(shape=(CHANNELS_GS, height, width))
    b = Convolution2D(num_channels, CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, border_mode='same')(a)
    c = Activation('relu')(b)
    d = c
    for i in range(num_res_blocks):
        d = resblock(d, num_channels)

    e = merge([c, d], mode='sum')
    f = Convolution2D(CHANNELS_GS, CONV_KERNEL_SIZE, CONV_KERNEL_SIZE, border_mode='same')(e)

    return Model(input=a, output=f)


# 5
def train_model(model, images, corruption_func, batch_size, samples_per_epoch, num_epochs, num_valid_samples):
    """
    Trains a given neural network Keras model.
    :param model: the network's Keras model
    :param images: list with paths of images to use as the dataset
    :param corruption_func: a function for corrupting the images
    :param batch_size: the size of the batch of examples for each iteration of SGD
    :param samples_per_epoch: number of samples used in each epoch
    :param num_epochs: number of epochs the optimization will use
    :param num_valid_samples: number of validation samples to test after each epoch
    """
    dataset_split_idx = int(len(images) * TRAINING_SET_SPLIT)
    training_images = images[: dataset_split_idx]
    validation_images = images[dataset_split_idx:]

    crop_size = model.input_shape[2: 4]
    training_set_gen = load_dataset(training_images, batch_size, corruption_func, crop_size)
    validation_set_gen = load_dataset(validation_images, batch_size, corruption_func, crop_size)

    adam = Adam(beta_2=BETA2_VAL)
    model.compile(loss='mean_squared_error', optimizer=adam)

    history = model.fit_generator(training_set_gen, samples_per_epoch=samples_per_epoch, nb_epoch=num_epochs,
                                  validation_data=validation_set_gen, nb_val_samples=num_valid_samples)


# 6
def restore_image(corrupted_image, base_model):
    """
    Restores a full image by using a trained neural network model.
    :param corrupted_image: a corrupted image in [0,1] range in of type float64
    :param base_model: a trained Keras model
    :return: a restored image
    """
    # adjusting the base model's size to the image's
    height, width = corrupted_image.shape
    a = Input(shape=(CHANNELS_GS, height, width))
    b = base_model(a)
    adjusted_model = Model(input=a, output=b)
    adjusted_model.set_weights(base_model.get_weights())

    # preparing image
    shifted_corrupted_im = corrupted_image - SUB_VALUE
    shifted_corrupted_im = shifted_corrupted_im[np.newaxis, ...]  # adding gs channel to image

    restored_image = adjusted_model.predict(shifted_corrupted_im[np.newaxis, ...])[0]

    # preparing results
    restored_image = restored_image[0].astype(np.float64)
    restored_image += SUB_VALUE
    restored_image = np.clip(restored_image, 0, 1)
    return restored_image


# 7.1.1
def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Adds gaussian noise to an image.
    :param image: a greyscale image with values in the [0, 1] range of type float64
    :param min_sigma: a non-negative scalar value representing the minimal variance of the gaussian distribution
    :param max_sigma: a non-negative scalar value larger than or equal to min_sigma,
                      representing the maximal variance of the gaussian distribution
    :return: the image corrupted by gaussian noise
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    rands = np.random.normal(0, sigma, image.shape)

    corrupted = image + rands
    corrupted = np.divide(np.round(corrupted * (MAX_VALUE - 1)), (MAX_VALUE - 1))
    corrupted = np.clip(corrupted, 0, 1)
    return corrupted


def train_model_quick_mode(model, images, corruption_func):
    """
    Trains a neural network for denoising\deblurring quickly and with fewer parameters.
    :param model: a Keras model of the neural network
    :param images: a list of file paths of images (grayscale, [0, 1], float64) for training & validation data
    :param corruption_func: a corruption function for the images
    """
    train_model(model, images, corruption_func, QUICK_BATCH_SIZE,
                QUICK_SAMPLES_PER_EPOCH, QUICK_EPOCHS, QUICK_VALID_SAMPLES)


# 7.1.2
def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    Trains a neural network for denoising images with iid gaussian blur.
    :param num_res_blocks: number of residual blocks in the network
    :param quick_mode: True trains the network faster but with lower parameters
    :return: a trained model
    """

    def corruption_func(im): return add_gaussian_noise(im, LDN_MIN_SIG, LDN_MAX_SIG)
    images = sol5_utils.images_for_denoising()
    model = build_nn_model(LDN_PATCH_SIZE, LDN_PATCH_SIZE, LDN_CHANNELS, num_res_blocks)

    if quick_mode:
        train_model_quick_mode(model, images, corruption_func)
    else:
        train_model(model, images, corruption_func, BATCH_SIZE,
                    SAMPLES_PER_EPOCH, LDN_EPOCHS, VALID_SAMPLES)

    return model


# 7.2
def add_motion_blur(image, kernel_size, angle):
    """
    Adds motion blur to an image.
    :param image: a grayscale image with values in the [0, 1] range of type float64
    :param kernel_size: an odd integer specifying the size of the kernel
    :param angle: an angle in radians in the range [0, pi)
    :return: the blurred image
    """
    kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    corrupted = convolve(image, kernel)
    return corrupted


# 7.2
def random_motion_blur(image, list_of_kernel_sizes):
    """
    Adds motion blur to an image using an angle in [0, pi) range and a random kernel size
    :param image: a grayscale image with values in the [0, 1] range of type float64
    :param list_of_kernel_sizes: a list of odd numbers representing kernel sizes
    :return: the blurred image
    """
    angle = np.random.uniform(0, np.pi)
    random_idx = np.random.randint(len(list_of_kernel_sizes))
    kernel_size = list_of_kernel_sizes[random_idx]

    corrupted = add_motion_blur(image, kernel_size, angle)
    return corrupted


# 7.2.2
def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    Trains a neural network for deblurring motion blurred images.
    :param num_res_blocks: number of residual blocks in the network
    :param quick_mode: True trains the network faster but with lower parameters
    :return: a trained model
    """

    def corruption_func(im): return random_motion_blur(im, [LDB_KERNEL_SIZE])
    images = sol5_utils.images_for_deblurring()
    model = build_nn_model(LDB_PATCH_SIZE, LDB_PATCH_SIZE, LDB_CHANNELS, num_res_blocks)

    if quick_mode:
        train_model_quick_mode(model, images, corruption_func)
    else:
        train_model(model, images, corruption_func, BATCH_SIZE,
                    SAMPLES_PER_EPOCH, LDB_EPOCHS, VALID_SAMPLES)

    return model
