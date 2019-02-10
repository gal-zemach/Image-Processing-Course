# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged 

import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass, map_coordinates, convolve
from scipy.misc import imsave as imsave
import shutil
import sol4_utils


DERIVATIVE_FILTER = np.array([1, 0, -1]).reshape(3, 1)
KERNEL_SIZE = 3
HCD_K = 0.04

SOC_N, SOC_M = 5, 5
SOC_RADIUS_FROM_EDGE = 7

SAMPLING_LEVEL = 3
SAMPLING_RADIUS = 3

RANSAC_SAMPLE_SIZE = 2
RANSAC_SAMPLE_SIZE_TRANS_ONLY = 1


def harris_corner_detector(im):
    """
    Detects harris corners.
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    dx = convolve(im, DERIVATIVE_FILTER)
    dy = convolve(im, DERIVATIVE_FILTER.transpose())

    ix_sqr = sol4_utils.blur_spatial(np.square(dx), KERNEL_SIZE)
    iy_sqr = sol4_utils.blur_spatial(np.square(dy), KERNEL_SIZE)
    ix_iy = sol4_utils.blur_spatial(np.multiply(dx, dy), KERNEL_SIZE)

    m1 = np.dstack([ix_sqr, ix_iy])
    m2 = np.dstack([ix_iy, iy_sqr])
    m = np.concatenate([m1[..., np.newaxis], m2[..., np.newaxis]], axis=3)

    response = np.linalg.det(m) - HCD_K * np.square(np.trace(m, axis1=2, axis2=3))
    bin_response = non_maximum_suppression(response)

    # preparing indices in [x,y] format
    idx = np.indices(im.shape)
    idx = np.dstack([idx[1], idx[0]])

    max_idx = idx[bin_response != 0]
    return max_idx


def transform_point(p, dest_level, src_level):
    """
    Transforms a point from a given level in a gaussian pyramid to it's coordinates in another.
    :param p: The point (x, y) coordinates.
    :param dest_level: an integer representing the level to transform the point to.
    :param src_level: an integer representing the level the point is in.
    :return: the (x, y) coordinates of the point in the destination level.
    """
    return (2 ** (src_level - dest_level)) * p


def window_size(radius):
    """
    Returns the size of a window with a given radius around center.
    (f.e. for radius=3, 7 will be returned as the window's size is 7*7)
    """
    return 2 * radius + 1


def get_range(points, radius):
    """
    Given a 1d array of points this returns the indices (not necessarily whole) of all points in a given radius around the point.
    :param points: 1d array of shape (n)
    :param radius: radius around the point to take
    :return: array of shape (n, 2*radius + 1) with the points in the radius around the given points.
    """
    start = points - radius
    stop = points + radius + 1
    k_range = np.linspace(0, 1, window_size(radius))
    return np.outer(stop - start, k_range) + start[:, np.newaxis]


def normalize_samples(samples):
    """
    Normalizes samples according to their mean.
    :param samples: An array of shape (n, k, k) containing n k*k samples
    :return: An array of the same size where the samples are normalized.
    """
    samples_mean = np.mean(samples, axis=(1, 2))
    samples_shifted = samples - samples_mean[:, np.newaxis, np.newaxis]
    samples_shifted_norm = np.linalg.norm(samples_shifted, axis=(1, 2))

    # turning zero division cases to 0 (as stated in the moodle)
    samples_shifted[samples_shifted_norm == 0] = np.zeros(samples_shifted.shape[1:3])
    samples_shifted_norm[samples_shifted_norm == 0] += 1

    samples_normalized = np.divide(samples_shifted, samples_shifted_norm[:, np.newaxis, np.newaxis])

    return samples_normalized


def sample_descriptor(im, corners, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param corners: An array with shape (N,2), where corners[i,:] are the [x,y] coordinates of the ith corner points.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """
    k = window_size(desc_rad)
    l3_corners = transform_point(corners, SAMPLING_LEVEL - 1, 0)

    row_range = get_range(l3_corners[:, 1], desc_rad)
    col_range = get_range(l3_corners[:, 0], desc_rad)

    row_range_2d = np.repeat(row_range[..., np.newaxis], k, axis=2)
    col_range_2d = np.repeat(col_range[:, np.newaxis, :], k, axis=1)

    samples = map_coordinates(im, [row_range_2d, col_range_2d], order=1, prefilter=False)
    samples_normalized = normalize_samples(samples)

    return samples_normalized


def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """
    corners = spread_out_corners(pyr[0], SOC_M, SOC_N, SOC_RADIUS_FROM_EDGE)
    desc = sample_descriptor(pyr[SAMPLING_LEVEL - 1], corners, SAMPLING_RADIUS)
    return corners, desc


def get_2_max_indices_in_rows(all_scores):
    """
    Finds the maximum 2 values in every row.
    :param all_scores: 2d array (n*m)
    :return: an array of shape (n*2) with the relevant indices
    """
    n1 = all_scores.shape[0]
    all_scores_copy = all_scores.copy()

    # small value to place in the rows' max cells when finding 2nd max
    dummy_min_val = np.amin(all_scores_copy) - 10000000

    max_scores_indices = np.argmax(all_scores_copy, axis=1).reshape(n1, 1)
    all_scores_copy[np.arange(n1).reshape(n1, 1), max_scores_indices] = dummy_min_val

    second_max_indices = np.argmax(all_scores_copy, axis=1).reshape(n1, 1)  # getting 2nd maximums
    return np.hstack((max_scores_indices, second_max_indices))


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """

    n1, n2, k = desc1.shape[0], desc2.shape[0], desc1.shape[1]
    desc1_flat = desc1.reshape(n1, k * k)
    desc2_flat = desc2.reshape(n2, k * k)

    all_scores = np.einsum('ij, kj->ik', desc1_flat, desc2_flat)

    # getting 2 max indices for every image
    max_indices_for_desc1_in_desc2 = get_2_max_indices_in_rows(all_scores)
    max_indices_for_desc2_in_desc1 = get_2_max_indices_in_rows(all_scores.transpose())

    # building array where all places indices that contain mutual matches will contain 2
    # scores that are smaller than min_score will be placed in the last row\col
    matches_count = np.zeros((n1 + 1, n2 + 1))
    max_indices_for_desc1_in_desc2[max_indices_for_desc1_in_desc2 < min_score] = n2
    max_indices_for_desc2_in_desc1[max_indices_for_desc2_in_desc1 < min_score] = n1

    matches_count[np.arange(n1).reshape(n1, 1), max_indices_for_desc1_in_desc2] += 1
    matches_count[max_indices_for_desc2_in_desc1, np.arange(n2).reshape(n2, 1)] += 1
    matches_count = matches_count[:n1, :n2]  # trimming last row\col

    idx = np.indices(matches_count.shape)
    idx = np.dstack([idx[0], idx[1]])

    matches = idx[matches_count == 2].astype(int)
    return matches[:, 0], matches[:, 1]


def reg_to_hom(points):
    """
    Converts point in [x, y] coordinates to homogenous coordinates
    :param points: an array of shape (n, 2)
    :return: array of shape (n, 3) with the points converted to homogenous ones
    """
    return np.hstack((points, np.ones((points.shape[0], 1))))


def hom_to_reg(points):
    """
    Converts point in homogenous coordinates to [x, y] coordinates
    :param points: an array of shape (n, 3)
    :return: array of shape (n, 2) with the points converted to regular coordinates
    """
    points_z = points[:, 2]
    reg_x = np.divide(points[:, 0], points_z)
    reg_y = np.divide(points[:, 1], points_z)
    return np.vstack((reg_x, reg_y)).transpose()


def left_multiply_vectors(mat, arr):
    """
    Left multiplies each vector in an array of vectors with a matrix.
    :param mat: array of shape (n * n)
    :param arr: array of shape (m * n)
    :return: array of shape (m * n) with the result
    """
    return np.einsum('ij, kj->ki', mat, arr)


def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """
    pos1_hom = reg_to_hom(pos1)
    pos2_hom = left_multiply_vectors(H12, pos1_hom)
    return hom_to_reg(pos2_hom)


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: False (default) will also compute rotation.
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """
    n = points1.shape[0]
    ordered_indices = np.arange(n)
    shuffled_indices = np.arange(n)
    best_inliers = np.array([])
    sample_size = RANSAC_SAMPLE_SIZE_TRANS_ONLY if translation_only else RANSAC_SAMPLE_SIZE

    for i in range(num_iter):
        np.random.shuffle(shuffled_indices)
        curr_indices = shuffled_indices[:sample_size]

        p1, p2 = points1[curr_indices, :], points2[curr_indices, :]
        H12 = estimate_rigid_transform(p1, p2, translation_only)
        points2_t = apply_homography(points1, H12)

        curr_dist = np.linalg.norm(points2 - points2_t, axis=1)
        curr_inliers = ordered_indices[curr_dist < inlier_tol]

        if curr_inliers.size > best_inliers.size:
            best_inliers = curr_inliers

    H = estimate_rigid_transform(points1[best_inliers, :], points2[best_inliers, :], translation_only)
    return H, best_inliers


def plot_lines(p1, p2, cond, color, line_width=.4):
    """
    Plots lines between two sets of points filtered by a boolean array.
    :param p1: set of n points in [x, y] format
    :param p2: set of n points in [x, y] format
    :param cond: boolean array of size n
    :param color: color for the lines as single character string
    :param line_width: width of the line to draw
    """
    p1_in = p1[cond]
    p2_in = p2[cond]
    for i in range(p1_in.shape[0]):
        x_s = [p1_in[i, 0], p2_in[i, 0]]
        y_s = [p1_in[i, 1], p2_in[i, 1]]
        plt.plot(x_s, y_s, color+"-", lw=line_width)


def display_matches(im1, im2, points1, points2, inliers):
    """
    Display matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :param points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param points2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    full_im = np.hstack((im1, im2))
    plt.imshow(full_im, cmap='gray')

    s_points2 = points2.copy()
    s_points2[:, 0] += im1.shape[1]

    # plotting points
    full_points = np.vstack((points1, s_points2))
    plt.plot(full_points[:, 0], full_points[:, 1], 'ro', markersize=1.5)

    # preparing inliers
    m = points1.shape[0]
    is_inlier = np.zeros(m).astype(bool)
    is_inlier[inliers] = True

    # plotting lines
    plot_lines(points1, s_points2, ~is_inlier, "b")
    plot_lines(points1, s_points2, is_inlier, "y", .6)

    plt.show()


def normalize_hom(H):
    """
    Normalizes a (3 * 3) homography to have 1 in the lower-left cell
    :param H: array of shape (3 * 3)
    :return: array of shape (3 * 3)
    """
    H /= H[2, 2]
    return H


def accumulate_homographies(H_successive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """
    H2m = [np.eye(3)]
    for i in range(m, 0, -1):
        new_H = np.dot(H2m[0], H_successive[i - 1])
        new_H = normalize_hom(new_H)
        H2m.insert(0, new_H)

    for i in range(m, len(H_successive)):
        new_H = np.dot(H2m[i], np.linalg.inv(H_successive[i]))
        new_H = normalize_hom(new_H)
        H2m.append(new_H)

    return H2m


def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the upper left corner,
     and the second row is the [x,y] of the lower right corner
    """
    w_i, h_i = w - 1, h - 1
    points = np.array([[0, 0], [w_i, 0], [0, h_i], [w_i, h_i]])

    points = reg_to_hom(points)
    new_points = left_multiply_vectors(homography, points)
    new_points = np.round(hom_to_reg(new_points)).astype(int)

    new_points = np.sort(new_points, axis=0)
    new_top_left = new_points[0]
    new_bottom_right = new_points[new_points.shape[0] - 1]

    return np.vstack((new_top_left, new_bottom_right))


def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homography.
    :return: A 2d warped image.
    """
    h, w = image.shape
    corners = compute_bounding_box(homography, w, h)

    x_range = np.arange(corners[0][0], corners[1][0] + 1)
    y_range = np.arange(corners[0][1], corners[1][1] + 1)
    x_i, y_i = np.meshgrid(x_range, y_range)
    rows, cols = x_i.shape

    grid = np.dstack((x_i, y_i)).reshape(rows * cols, 2)
    inv_hom = np.linalg.inv(homography)
    grid_t = apply_homography(grid, inv_hom).reshape(rows, cols, 2)

    x_i_t, y_i_t = grid_t[:, :, 0], grid_t[:, :, 1]

    warped_image = map_coordinates(image, [y_i_t, x_i_t], order=1, prefilter=False)
    return warped_image


def warp_image(image, homography):
    """
    Warps an RGB image with a given homography.
    :param image: an RGB image.
    :param homography: homograhpy.
    :return: A warped image.
    """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
    Filters rigid transformations encoded as homographies by the amount of translation from left to right.
    :param homographies: homograhpies to filter.
    :param minimum_right_translation: amount of translation below which the transformation is discarded.
    :return: filtered homographies..
    """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
    Computes rigid transforming points1 towards points2, using least squares method.
    points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
    :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
    :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
    :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
    :return: A 3x3 array with the computed homography.
    """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
    Finds local maximas of an image.
    :param image: A 2D array representing an image.
    :return: A boolean array with the same shape as the input image, where True indicates local maximum.
    """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
    Splits the image im to m by n rectangles and uses harris_corner_detector on each.
    :param im: A 2D array representing an image.
    :param m: Vertical number of rectangles.
    :param n: Horizontal number of rectangles.
    :param radius: Minimal distance of corner points from the boundary of the image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
    Generates panorama from a set of images.
    """

    def __init__(self, data_dir, file_prefix, num_images):
        """
        The naming convention for a sequence of images is file_prefixN.jpg,
        where N is a running number 001, 002, 003...
        :param data_dir: path to input images.
        :param file_prefix: see above.
        :param num_images: number of images to produce the panoramas with.
        """
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
        compute homographies between all images to a common coordinate system
        :param translation_only: see estimte_rigid_transform
        """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
        """
        combine slices from input images to panoramas.
        :param number_of_panoramas: how many different slices to take from each input image
        """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
