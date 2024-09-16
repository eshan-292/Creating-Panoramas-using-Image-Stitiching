from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
# from cv2 import subtract


import cv2
import random
import numpy as np



def resize(image, shape, interpolation):
    # Compute scale factors for resizing
    scale_x = shape[1] / image.shape[1]
    scale_y = shape[0] / image.shape[0]

    # Determine interpolation method
    if interpolation == 'INTER_NEAREST' or 0:
        method = 'nearest'
    elif interpolation == 'INTER_LINEAR'or 1:
        method = 'linear'
    else:
        raise ValueError("Unsupported interpolation method")

    # Generate new indices
    rows, cols = np.indices(shape)
    rows = rows / scale_y
    cols = cols / scale_x

    # Apply interpolation
    if method == 'nearest':
        rows = np.round(rows).astype(int)
        cols = np.round(cols).astype(int)
    elif method == 'linear':
        rows_floor = np.floor(rows).astype(int)
        cols_floor = np.floor(cols).astype(int)
        rows_ceil = np.ceil(rows).astype(int)
        cols_ceil = np.ceil(cols).astype(int)

        rows_frac = rows - rows_floor
        cols_frac = cols - cols_floor

        # Clip indices to image boundaries
        rows_floor = np.clip(rows_floor, 0, image.shape[0] - 1)
        cols_floor = np.clip(cols_floor, 0, image.shape[1] - 1)
        rows_ceil = np.clip(rows_ceil, 0, image.shape[0] - 1)
        cols_ceil = np.clip(cols_ceil, 0, image.shape[1] - 1)

        # Compute pixel values using bilinear interpolation
        interpolated_values = (1 - rows_frac) * (1 - cols_frac) * image[rows_floor, cols_floor] + \
                              (1 - rows_frac) * cols_frac * image[rows_floor, cols_ceil] + \
                              rows_frac * (1 - cols_frac) * image[rows_ceil, cols_floor] + \
                              rows_frac * cols_frac * image[rows_ceil, cols_ceil]

        return interpolated_values.astype(np.uint8)

def GaussianBlur(image, sigma):
    # Define kernel size based on sigma
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create 1D Gaussian kernel
    kernel1d = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    kernel1d = np.exp(-kernel1d ** 2 / (2 * sigma ** 2))
    kernel1d /= kernel1d.sum()

    # Perform 2D convolution with separable kernel
    blurred_image = np.apply_along_axis(lambda x: np.convolve(x, kernel1d, mode='same'), axis=0, arr=image)
    blurred_image = np.apply_along_axis(lambda x: np.convolve(x, kernel1d, mode='same'), axis=1, arr=blurred_image)

    return blurred_image.astype(np.uint8)

def subtract(image1, image2):
    # Convert images to float32 for subtraction
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    # Perform element-wise subtraction
    result = np.clip(image1 - image2, 0, 255)

    return result.astype(np.uint8)




def gen_base_img(image, sigma, assumed_blur):
    
    # image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=1)
    image = resize(image, (image.shape[0]*2, image.shape[1]*2), 'INTER_LINEAR')
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    # return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur
    return GaussianBlur(image, sigma_diff)

def calc_octaves(image_shape):
    return int(round(log(min(image_shape)) / log(2) - 1))


def genDoG(gaussian_pyramid):
    dog_pyramid = []

    for gaussian_octave in gaussian_pyramid:
        dog_octave = []
        for prev_img, curr_img in zip(gaussian_octave, gaussian_octave[1:]):
            dog_octave.append(subtract(curr_img - prev_img))
        dog_pyramid.append(dog_octave)
    
    return np.array(dog_pyramid, dtype=object)




def gen_kernels(sigma, num_intervals):
    
    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = np.zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = sigma

    for image_idx in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_idx - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_idx] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
        
    return gaussian_kernels

def gen_gaussian(base_image, num_octaves, gaussian_kernels):

    gaussian_pyramid = []

    for octave_idx in range(num_octaves):
        octave_images = []
        octave_images.append(base_image)   # first image in each octave is the original image

        for kernel in gaussian_kernels[1:]:
            blurred_image = GaussianBlur(octave_images[-1], kernel)
            octave_images.append(blurred_image)
        gaussian_pyramid.append(octave_images)
        
        octave_base = octave_images[-3]
        resized_image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=0)
        base_image = resized_image

    return np.array(gaussian_pyramid, dtype=object)


def localise_extremum(i, j, image_index, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width, eigenvalue_ratio=10, num_attempts_until_convergence=5):
    
    # Check if the pixel is an extremum in the 3x3x3 pixel cube
    extremum_is_outside_image = False
    image_shape = dog_images_in_octave[0].shape
    for attempt_index in range(num_attempts_until_convergence):
        
        first_image, second_image, third_image = dog_images_in_octave[image_index-1:image_index+2]
        pixel_cube = stack([first_image[i-1:i+2, j-1:j+2],
                            second_image[i-1:i+2, j-1:j+2],
                            third_image[i-1:i+2, j-1:j+2]]).astype('float32') / 255.
        hessian = calc_hessian(pixel_cube)
        gradient = calc_grad(pixel_cube)
        
        # Solve the linear system
        extremum_update = -lstsq(hessian, gradient, rcond=None)[0]
        if abs(extremum_update[0]) < 0.5 and abs(extremum_update[1]) < 0.5 and abs(extremum_update[2]) < 0.5:
            break
        j += int(round(extremum_update[0]))
        i += int(round(extremum_update[1]))
        image_index += int(round(extremum_update[2]))
        
        # check if the new pixel is within the image
        if i < image_border_width or i >= image_shape[0] - image_border_width or j < image_border_width or j >= image_shape[1] - image_border_width or image_index < 1 or image_index > num_intervals:
            extremum_is_outside_image = True
            break
    if extremum_is_outside_image:
        return None
    if attempt_index >= num_attempts_until_convergence - 1:
        
        return None
    functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * dot(gradient, extremum_update)
    # Check if the extremum is a contrast maximum
    if abs(functionValueAtUpdatedExtremum) * num_intervals >= contrast_threshold:
        xy_hessian = hessian[:2, :2]
        xy_hessian_trace = trace(xy_hessian)
        xy_hessian_det = det(xy_hessian)
        #  Check if the eigenvalues of the Hessian matrix at the extremum are all positive
        if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
            
            keypoint = cv2.KeyPoint()
            keypoint.pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index))
            keypoint.octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
            keypoint.size = sigma * (2 ** ((image_index + extremum_update[2]) / float32(num_intervals))) * (2 ** (octave_index + 1))  # octave_index + 1 because the input image was doubled
            keypoint.response = abs(functionValueAtUpdatedExtremum)
            return keypoint, image_index
    return None



def calc_hessian(pixel_array):
    center_pixel_value = pixel_array[1, 1, 1]
    
    dss = pixel_array[0, 1, 1] - 2 * center_pixel_value + pixel_array[2, 1, 1]
    dxx = pixel_array[1, 1, 0] - 2 * center_pixel_value + pixel_array[1, 1, 2]
    dyy = pixel_array[1, 0, 1] - 2 * center_pixel_value + pixel_array[1, 2, 1]
    
    dys = 0.25 * (pixel_array[0, 0, 1] + pixel_array[2, 2, 1] - pixel_array[0, 2, 1] - pixel_array[2, 0, 1])
    dxy = 0.25 * (pixel_array[1, 0, 0] + pixel_array[1, 2, 2] - pixel_array[1, 0, 2] - pixel_array[1, 2, 0])
    dxs = 0.25 * (pixel_array[0, 1, 0] + pixel_array[2, 1, 2] - pixel_array[0, 1, 2] - pixel_array[2, 1, 0])
    
    
    return np.array([[dxx, dxy, dxs], 
                     [dxy, dyy, dys],
                     [dxs, dys, dss]])



def check_pixel_extrema(first_subimage, second_subimage, third_subimage, threshold):
    
    center_pixel_value = second_subimage[1, 1]
    if np.abs(center_pixel_value) <= threshold:
        return False
    
    #  Check if the center pixel is the maximum or minimum in the 3x3x3 pixel cube
    if center_pixel_value > 0:
        return (center_pixel_value >= first_subimage).all() and \
               (center_pixel_value >= third_subimage).all() and \
               (center_pixel_value >= second_subimage[0, :]).all() and \
               (center_pixel_value >= second_subimage[2, :]).all() and \
               center_pixel_value >= second_subimage[1, 0] and \
               center_pixel_value >= second_subimage[1, 2]
    
    if center_pixel_value < 0:
        return (center_pixel_value <= first_subimage).all() and \
               (center_pixel_value <= third_subimage).all() and \
               (center_pixel_value <= second_subimage[0, :]).all() and \
               (center_pixel_value <= second_subimage[2, :]).all() and \
               center_pixel_value <= second_subimage[1, 0] and \
               center_pixel_value <= second_subimage[1, 2]
    
def calc_grad(pixel_array):
    ds = 0.5 * (pixel_array[2, 1, 1] - pixel_array[0, 1, 1])
    dx = 0.5 * (pixel_array[1, 1, 2] - pixel_array[1, 1, 0])
    dy = 0.5 * (pixel_array[1, 2, 1] - pixel_array[1, 0, 1])
    
    return array([dx, dy, ds])




def find_ss_extrema(gaussian_pyramid, dog_pyramid, num_intervals, sigma, border_width, contrast_threshold=0.04):
    threshold = int(0.5 * contrast_threshold / num_intervals * 255)  # From OpenCV implementation
    keypoints = []

    # Iterate over the DoG pyramid and find the extrema in each 3x3x3 pixel cube
    for octave_idx, dog_images_octave in enumerate(dog_pyramid):
        for img_idx, (prev_img, curr_img, next_img) in enumerate(zip(dog_images_octave, dog_images_octave[1:], dog_images_octave[2:])):
            for i in range(border_width, prev_img.shape[0] - border_width):
                for j in range(border_width, prev_img.shape[1] - border_width):
                    if check_pixel_extrema(prev_img[i-1:i+2, j-1:j+2], curr_img[i-1:i+2, j-1:j+2], next_img[i-1:i+2, j-1:j+2], threshold):
                        localization_result = localise_extremum(i, j, img_idx + 1, octave_idx, num_intervals, dog_images_octave, sigma, contrast_threshold, border_width)
                        if localization_result is not None:
                            keypoint, localized_img_idx = localization_result
                            keypoints_with_orientations = comp_keypoints(keypoint, octave_idx, gaussian_pyramid[octave_idx][localized_img_idx])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                keypoints.append(keypoint_with_orientation)
    return keypoints




# Keypoints



# def comp_keypts(keypoint1, keypoint2):
    
#     if keypoint1.pt[0] != keypoint2.pt[0]:
#         return keypoint1.pt[0] - keypoint2.pt[0]
#     if keypoint1.pt[1] != keypoint2.pt[1]:
#         return keypoint1.pt[1] - keypoint2.pt[1]
#     if keypoint1.size != keypoint2.size:
#         return keypoint2.size - keypoint1.size
#     if keypoint1.angle != keypoint2.angle:
#         return keypoint1.angle - keypoint2.angle
#     if keypoint1.response != keypoint2.response:
#         return keypoint2.response - keypoint1.response
#     if keypoint1.octave != keypoint2.octave:
#         return keypoint2.octave - keypoint1.octave
#     return keypoint2.class_id - keypoint1.class_id


def cmp_to_key(mycmp):
    
    class K(object):
        __slots__ = ['obj']
        def __init__(self, obj):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        __hash__ = None
    return K

def comp_keypts(keypoint1, keypoint2):
    
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    
    return keypoint2.class_id - keypoint1.class_id





def conv_keypoints(keypoints):
    
    converted_keypoints = []
    # Convert the keypoint coordinates and size from floating-point to fixed-point
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints

# Descriptors Generation

def octave_decode(keypoint):
    
    # Decode the keypoint's octave to extract the image index and the scale
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / float32(1 << octave) if octave >= 0 else float32(1 << -octave)
    return octave, layer, scale


def rm_duplicates(keypoints):
    
    if len(keypoints) < 2:
        return keypoints

    # Sort keypoints lexicographically
    keypoints.sort(key=cmp_to_key(comp_keypts))

    unique_keypoints = [keypoints[0]]

#   # Remove duplicate keypoints
    for next_keypoint in keypoints[1:]:
        last_unique_keypoint = unique_keypoints[-1]
        if (last_unique_keypoint.pt[0] != next_keypoint.pt[0] or
            last_unique_keypoint.pt[1] != next_keypoint.pt[1] or
            last_unique_keypoint.size != next_keypoint.size or
            last_unique_keypoint.angle != next_keypoint.angle):
            unique_keypoints.append(next_keypoint)
    return unique_keypoints



def comp_keypoints(keypoint, octave_index, gaussian_image, radius_factor=3, num_bins=36, peak_ratio=0.8, scale_factor=1.5):
    keypoints_with_orientations = []
    image_shape = gaussian_image.shape

    # calculate gradient magnitude and orientation at each pixel   
    scale = scale_factor * keypoint.size / (2 ** (octave_index + 1))  # compare with keypoint.size computation in localise_extremum()
    radius = int(round(radius_factor * scale))
    weight_factor = -0.5 / (scale ** 2)
    raw_histogram = np.zeros(num_bins)
    smooth_histogram = np.zeros(num_bins)

#   3x3 region around the keypoint
    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / (2 ** octave_index))) + i
        if 0 < region_y < image_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / (2 ** octave_index))) + j
                if 0 < region_x < image_shape[1] - 1:
                    dx = gaussian_image[region_y, region_x + 1] - gaussian_image[region_y, region_x - 1]
                    dy = gaussian_image[region_y - 1, region_x] - gaussian_image[region_y + 1, region_x]
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))  # constant in front of exponential can be dropped because we will find peaks later
                    histogram_index = int(round(gradient_orientation * num_bins / 360.))
                    raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

    # compute smoothed orientation histogram using a 16-bin Gaussian window
    for n in range(num_bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.

    orientation_max = np.max(smooth_histogram)
    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]

    # find the peaks in the orientation histogram
    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            
            # Quadratic peak interpolation
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            orientation = 360. - interpolated_peak_index * 360. / num_bins
            if abs(orientation - 360.) < 1e-7:
                orientation = 0
            
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    
    return keypoints_with_orientations



def gen_desc(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    descriptors = []
    
    # Iterate over all keypoints
    for keypoint in keypoints:
        octave, layer, scale = octave_decode(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        point = np.round(scale * np.array(keypoint.pt)).astype(int)
        
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        
        # Calculate the descriptor for this keypoint
        col_bin_list = []
        magnitude_list = []
        row_bin_list = []
        orientation_bin_list = []
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))   

        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(np.round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))  
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))     


        # Compute the weighted orientation magnitude and the corresponding histogram bin
        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if 0 < row_bin < window_width and 0 < col_bin < window_width:
                    window_row = int(np.round(point[1] + row))
                    window_col = int(np.round(point[0] + col))
                    if 0 < window_row < num_rows - 1 and 0 < window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        col_bin_list.append(col_bin)
                        row_bin_list.append(row_bin)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)
                        magnitude_list.append(weight * gradient_magnitude)
                        

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            row_bin_floor, col_bin_floor, orientation_bin_floor = int(np.floor(row_bin)), int(np.floor(col_bin)), int(np.floor(orientation_bin))
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111
            
            
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), 1e-7)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)

    return np.array(descriptors, dtype='float32')



def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    
    image = image.astype('float32')
    base_image = gen_base_im(image, sigma, assumed_blur)
    num_octaves = calc_octaves(base_image.shape)
    gaussian_kernels = gen_kernels(sigma, num_intervals)
    gaussian_images = gengaussian(base_image, num_octaves, gaussian_kernels)
    dog_images = genDOG(gaussian_images)
    keypoints = find_ss_extrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)
    keypoints = rm_duplicates(keypoints)
    keypoints = conv_keypoints(keypoints)
    descriptors = gen_desc(keypoints, gaussian_images)
    return keypoints, descriptors

# Other Temporary Functions

def gen_base_im(image, sigma, assumed_blur):
    
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=1)
    # image = resize(image, (image.shape[0]*2, image.shape[1]*2), 'INTER_LINEAR')
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur


def gengaussian(image, num_octaves, gaussian_kernels):
   
    gaussian_images = []

    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(image)  # first image in octave already has the correct blur
        for gaussian_kernel in gaussian_kernels[1:]:
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[-3]
        image = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=0)
        # image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=0)
    return array(gaussian_images, dtype=object)

def genDOG(gaussian_images):
    """Generate Difference-of-Gaussians image pyramid
    """
    
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(cv2.subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
        dog_images.append(dog_images_in_octave)
    return array(dog_images, dtype=object)

# Testing the SIFT implementation
def test_sift():
    # Load image
    image = cv2.imread('data/Images/Office/1.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create SIFT object
    sift = cv2.SIFT_create()

    # Detect SIFT keypoints
    keypoints = sift.detect(gray, None)

    # Draw keypoints
    image = cv2.drawKeypoints(gray, keypoints, image)
    # print the number of keypoints detected
    print('Number of keypoints:', len(keypoints))

    # Display image
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #  Use custom SIFT implementation
    keypoints, descriptors = computeKeypointsAndDescriptors(gray)
    print('Number of keypoints:', len(keypoints))
    print('Descriptors shape:', descriptors.shape)
    # show the keypoints
    # image = cv2.drawKeypoints(gray, keypoints, image)
    # cv2.imshow('Image_custom', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# test_sift()





class Match:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance

def brute_force_matcher(descriptors1, descriptors2, threshold=0.9):
    """Brute-force matcher using SIFT descriptors"""
    matches = []
    for i in range(len(descriptors1)):
        descriptor1 = descriptors1[i]
        best_distance = float('inf')
        best_match_index = -1
        second_best_distance = float('inf')
        for j in range(len(descriptors2)):
            descriptor2 = descriptors2[j]
            distance = np.linalg.norm(descriptor1 - descriptor2)
            if distance < best_distance:
                second_best_distance = best_distance
                best_distance = distance
                best_match_index = j
            elif distance < second_best_distance:
                second_best_distance = distance
        # if best_distance / second_best_distance < threshold:
        matches.append(Match(i, best_match_index, best_distance))
    return matches


def knn_matcher(descriptors1, descriptors2, k=2, threshold=0.7):
    """KNN matcher using SIFT descriptors"""
    matches = []
    for i in range(len(descriptors1)):
        descriptor1 = descriptors1[i]
        distances = np.linalg.norm(descriptors2 - descriptor1, axis=1)
        sorted_indices = np.argsort(distances)
        best_matches = []
        for j in range(k):
            if distances[sorted_indices[j]] < threshold * distances[sorted_indices[j + 1]]:
                best_matches.append(Match(i, sorted_indices[j], distances[sorted_indices[j]]))

        # sort the best matches according to the distance
        best_matches = sorted(best_matches, key=lambda match: match.distance)
        matches.append(best_matches)
    return matches






# Computers a homography from 4-correspondences

def calculateHomography(correspondences):
    #loop through correspondences and create assemble matrix
    aList = []
    for corr in correspondences:
        p1 = np.matrix([corr.item(0), corr.item(1), 1])
        p2 = np.matrix([corr.item(2), corr.item(3), 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
              p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
              p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        aList.append(a1)
        aList.append(a2)

    matrixA = np.matrix(aList)

    #svd composition
    u, s, v = np.linalg.svd(matrixA)

    #reshape the min singular value into a 3 by 3 matrix
    h = np.reshape(v[8], (3, 3))

    #normalize and now we have h
    h = (1/h.item(8)) * h
    return h


#
#Calculate the geometric distance between estimated points and original points
#
def geometricDistance(correspondence, h):

    p1 = np.transpose(np.matrix([correspondence[0].item(0), correspondence[0].item(1), 1]))
    estimatep2 = np.dot(h, p1)
    # check for divide by zero
    if estimatep2.item(2) == 0:
        return float('inf')
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[0].item(2), correspondence[0].item(3), 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)



# Runs through ransac algorithm, creating homographies from random correspondences

def ransac(corr, thresh):
    maxInliers = []
    finalH = None
    for i in range(10000):
        #find 4 random points to calculate a homography
        corr1 = corr[np.random.randint(0, len(corr))]
        corr2 = corr[np.random.randint(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[np.random.randint(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[np.random.randint(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        #call the homography function on those points
        h = calculateHomography(randomFour)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i], h)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h
        # print ("Corr size: ", len(corr), " NumInliers: ", len(inliers), "Max inliers: ", len(maxInliers))

        if len(maxInliers) > (len(corr)*thresh):
            break
    return finalH, maxInliers



def warpPerspective(image, H, output_shape):
    """
    Apply a perspective transformation to an image using a given homography matrix.

    Args:
        image: Input image, numpy array of shape (H, W, C).
        H: Homography matrix, numpy array of shape (3, 3).
        output_shape: Shape of the output image (height, width).

    Returns:
        output_image: Warped output image, numpy array of shape (output_height, output_width, C).
    """
    output_height, output_width = output_shape
    channels = image.shape[2] if len(image.shape) == 3 else 1

    # Create output image
    output_image = np.zeros((output_height, output_width, channels), dtype=image.dtype)

    # Calculate inverse of H for backward mapping
    H_inv = np.linalg.inv(H)

    # Iterate over each pixel in the output image
    for y_out in range(output_height):
        for x_out in range(output_width):
            # Map output pixel to input pixel using inverse homography
            p_out = np.array([x_out, y_out, 1]).reshape(3, 1)
            p_in = np.dot(H_inv, p_out)
            p_in /= p_in[2]  # Normalize homogeneous coordinates

            x_in, y_in = p_in[0, 0], p_in[1, 0]

            # Check if the mapped point is within the input image bounds
            if 0 <= x_in < image.shape[1] and 0 <= y_in < image.shape[0]:
                # Bilinear interpolation
                x_in_floor, y_in_floor = int(np.floor(x_in)), int(np.floor(y_in))
                x_in_ceil, y_in_ceil = min(x_in_floor + 1, image.shape[1] - 1), min(y_in_floor + 1, image.shape[0] - 1)

                # Calculate interpolation weights
                dx, dy = x_in - x_in_floor, y_in - y_in_floor

                # Interpolate each channel
                for c in range(channels):
                    interp_val = (1 - dy) * ((1 - dx) * image[y_in_floor, x_in_floor, c] +
                                             dx * image[y_in_floor, x_in_ceil, c]) + \
                                 dy * ((1 - dx) * image[y_in_ceil, x_in_floor, c] +
                                       dx * image[y_in_ceil, x_in_ceil, c])

                    output_image[y_out, x_out, c] = interp_val

    return output_image



def detect_and_match_features_lin(image1, image2, direction):
    

    # use custom SIFT
    # convert the images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    keypoints1, descriptors1 = computeKeypointsAndDescriptors(image1_gray)
    keypoints2, descriptors2 = computeKeypointsAndDescriptors(image2_gray)

    # # load the keypoints and descriptors
    # keypoints1 = pickle.load(open("data/keypoints1.pkl", "rb"))
    # descriptors1 = pickle.load(open("data/descriptors1.pkl", "rb"))
    # keypoints2 = pickle.load(open("data/keypoints2.pkl", "rb"))
    # descriptors2 = pickle.load(open("data/descriptors2.pkl", "rb"))
    # # convert the keypoints back to original format
    # keypoints1 = [(x, y, size, angle, response, octave, class_id) for x, y, size, angle, response, octave, class_id in keypoints1]
    # keypoints2 = [(x, y, size, angle, response, octave, class_id) for x, y, size, angle, response, octave, class_id in keypoints2]



    # # Detect keypoints and compute descriptors using SIFT
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    
    # convert keypoints to tuple
    keypoints1 = [(keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id) for keypoint in keypoints1]
    keypoints2 = [(keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id) for keypoint in keypoints2]


   
    # bf = cv2.BFMatcher()
    # matches = bf.match(descriptors1, descriptors2)
    # matches = sorted(matches, key = lambda x:x.distance)

    # # print the length of the matches
    # print("Number of matches:", len(matches))

    # new_matches = []
    # # remove the matches such that the angle wrt x axis is greater than 30 degrees
    # for match in matches:
    #     x1, y1 = keypoints1[match.queryIdx][:2]
    #     x2, y2 = keypoints2[match.trainIdx][:2]

    #     angle = np.arctan((y2 - y1) / (x2 - x1))
    #     angle = np.degrees(angle)
    #     if abs(angle) < 20:
    #         new_matches.append(match)
        
    # sorted(new_matches, key = lambda x:x.distance)



    matches = brute_force_matcher(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda match: match.distance)
    print("Number of matches:", len(matches))
    new_matches = []
    # remove the matches such that the angle wrt x axis is greater than 30 degrees
    for match in matches:
        x1, y1 = keypoints1[match.queryIdx][:2]
        x2, y2 = keypoints2[match.trainIdx][:2]
        angle = np.arctan((y2 - y1) / (x2 - x1))
        angle = np.degrees(angle)
        if abs(angle) < 20:
            new_matches.append(match)

    # sort the matches
    new_matches = sorted(new_matches, key=lambda match: match.distance)

    
    
    
    good_matches = new_matches[:min(1000, len(new_matches))]
    # src_pts = np.array([keypoints1[match.queryIdx][:2] for match in good_matches])
    # dst_pts = np.array([keypoints2[match.trainIdx][:2] for match in good_matches])



    # if the direction is left, then remove all the matches that are not on the left side of the image
    if direction == 'left':
        center = image1.shape[1] // 2
        # remove all the matches that are on the right side of the image
        good_matches = [m for m in good_matches if keypoints1[m.queryIdx][0] <= center]
    elif direction == 'right':
        center = image1.shape[1] // 2
        # remove all the matches that are on the left side of the image
        good_matches = [m for m in good_matches if keypoints1[m.queryIdx][0] >= center]
        
    
# print number of final matches
    print("Number of final matches:", len(good_matches))

    correspondences = []
    for match in good_matches:
        x1, y1 = keypoints1[match.queryIdx][:2]
        x2, y2 = keypoints2[match.trainIdx][:2]
        
        # correspondences.append([x1, y1, x2, y2])
        correspondences.append([x2, y2, x1, y1])

    correspondences = np.matrix(correspondences)
    


        
    
    
    
    # convert the keypoints to the cv2 keypoints for visualization
    # keypoints1 = [cv2.KeyPoint(x, y, size, angle, response, int(octave), int(class_id)) for x, y, size, angle, response, octave, class_id in keypoints1]
    # keypoints2 = [cv2.KeyPoint(x, y, size, angle, response, int(octave), int(class_id)) for x, y, size, angle, response, octave, class_id in keypoints2]
    # convert the good matches to the cv2 matches
    # good_matches = [cv2.DMatch(m.queryIdx, m.trainIdx, m.distance) for m in good_matches]   
    
    # # Show the matched keypoints
    # matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("Matched Keypoints", matched_image)
    # cv2.waitKey(0)
    
    
    # run ransac
    H_custom, inliers = ransac(correspondences, 5)
    # convert to numpy.ndarray
    H_custom = np.array(H_custom)
    
    # # print H_custom
    # print("Homography CUstom:", H_custom)

    # Extract matched keypoints
    src_pts = np.float32([keypoints1[m.queryIdx][:2] for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx][:2] for m in good_matches]).reshape(-1, 1, 2)
    
    
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    # print H
    # print("Homography OpenCV:", H)

    return H
    



def get_new_frame_size_and_homography_lin(image2, image1, homography):
    # reading the size of the  image
    h1, w1 = image2.shape[:2]

    
    intial_coordinates = np.array([[0, w1-1, w1-1, 0], [0, 0, h1-1, h1-1], [1, 1, 1, 1]])

    final_coordinates = np.dot(homography, intial_coordinates)

    [x, y, z] = final_coordinates
    x = np.divide(x, z)
    y = np.divide(y, z)

    # Finding the dimentions of the stitched image frame and the "Correction" factor
    # to correct the homography matrix
    x_min = int(round(min(x)))
    x_max = int(round(max(x)))
    y_min = int(round(min(y)))
    y_max = int(round(max(y)))

    new_width = x_max
    new_height = y_max
    correction = [0,0]
    if x_min < 0:
        new_width  -= x_min
        correction[0] = -x_min
    if y_min < 0:
        new_height -= y_min
        correction[1] = -y_min

    # Again correcting New_Width and New_Height
    # Helpful when secondary image is overlaped on the left hand side of the Base image
    if new_width < image1.shape[1] + correction[0]:
        new_width = image1.shape[1] + correction[0]
    if new_height < image1.shape[0] + correction[1]:
        new_height = image1.shape[0] + correction[1]

    # finding the coordinates of the corners of the image if they were all were within the frame.
    x = np.add(x, correction[0])
    y = np.add(y, correction[1])
    old_points = np.float32([[0, 0], [w1-1, 0], [w1-1, h1-1], [0, h1-1]])

    final_points = np.float32(np.array([x, y]).T)

    # updating the homography matrix
    homography = cv2.getPerspectiveTransform(old_points, final_points)

    return [int(new_height), int(new_width)] , correction, homography
        
        
        





# # Testing the sift algorithm
# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import time
# import os

# # # Load the image
# # image = cv2.imread('data/Images/Office/1.jpg')

# # # Compute the keypoints and descriptors
# # start_time = time.time()
# # keypoints, descriptors = computeKeypointsAndDescriptors(image)
# # end_time = time.time()
# # print('Elapsed time: %.2f seconds' % (end_time - start_time))

# # # # Display the keypoints
# # # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
# # # for keypoint in keypoints:
# # #     x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
# # #     size = int(keypoint.size)
# # #     cv2.circle(image, (x, y), size, (0, 255, 0))
# # #     cv2.line(image, (x, y), (x + int(size * np.cos(np.deg2rad(keypoint.angle))), y + int(size * np.sin(np.deg2rad(keypoint.angle)))), (0, 255, 0))
# # # plt.imshow(image)
# # # plt.axis('off')
# # # plt.show()










# # # ############################  
# # # Compute the matches
# # image1 = cv2.imread('data/Images/Office/1.jpg')
# # image2 = cv2.imread('data/Images/Office/2.jpg')
# # image3 = cv2.imread('data/Images/Office/3.jpg')
# # image4 = cv2.imread('data/Images/Office/4.jpg')
# # image5 = cv2.imread('data/Images/Office/5.jpg')
# # image6 = cv2.imread('data/Images/Office/6.jpg')

# # # convert to grayscale
# # image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# # image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# # image3_gray = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
# # image4_gray = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
# # image5_gray = cv2.cvtColor(image5, cv2.COLOR_BGR2GRAY)
# # image6_gray = cv2.cvtColor(image6, cv2.COLOR_BGR2GRAY)


# # keypoints1, descriptors1 = computeKeypointsAndDescriptors(image1_gray)
# # keypoints2, descriptors2 = computeKeypointsAndDescriptors(image2_gray)
# # keypoints3, descriptors3 = computeKeypointsAndDescriptors(image3_gray)
# # keypoints4, descriptors4 = computeKeypointsAndDescriptors(image4_gray)
# # keypoints5, descriptors5 = computeKeypointsAndDescriptors(image5_gray)
# # keypoints6, descriptors6 = computeKeypointsAndDescriptors(image6_gray)

# # # save the descriptors
# # pickle.dump(descriptors1, open('office_1_descriptors.pkl', 'wb'))
# # pickle.dump(descriptors2, open('office_2_descriptors.pkl', 'wb'))
# # pickle.dump(descriptors3, open('office_3_descriptors.pkl', 'wb'))
# # pickle.dump(descriptors4, open('office_4_descriptors.pkl', 'wb'))
# # pickle.dump(descriptors5, open('office_5_descriptors.pkl', 'wb'))
# # pickle.dump(descriptors6, open('office_6_descriptors.pkl', 'wb'))



# # # convert the keypoints to numpy array
# # keypoints1 = np.array([(keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id) for keypoint in keypoints1]) 
# # keypoints2 = np.array([(keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id) for keypoint in keypoints2])
# # keypoints3 = np.array([(keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id) for keypoint in keypoints3])
# # keypoints4 = np.array([(keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id) for keypoint in keypoints4])
# # keypoints5 = np.array([(keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id) for keypoint in keypoints5])
# # keypoints6 = np.array([(keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id) for keypoint in keypoints6])



# # # save the keypoints
# # pickle.dump(keypoints1, open('office_1_keypoints.pkl', 'wb'))
# # pickle.dump(keypoints2, open('office_2_keypoints.pkl', 'wb'))
# # pickle.dump(keypoints3, open('office_3_keypoints.pkl', 'wb'))
# # pickle.dump(keypoints4, open('office_4_keypoints.pkl', 'wb'))
# # pickle.dump(keypoints5, open('office_5_keypoints.pkl', 'wb'))
# # pickle.dump(keypoints6, open('office_6_keypoints.pkl', 'wb'))




# image1 = cv2.imread('data/Images/Office/1.jpg')
# image2 = cv2.imread('data/Images/Office/2.jpg')



# # Load the descriptors
# descriptors1 = pickle.load(open('data/descriptors1.pkl', 'rb'))
# descriptors2 = pickle.load(open('data/descriptors2.pkl', 'rb'))

# # Load the keypoints
# keypoints1 = pickle.load(open('data/keypoints1.pkl', 'rb'))
# keypoints2 = pickle.load(open('data/keypoints2.pkl', 'rb'))

# # convert the keypoints back to the cv2 keypoints
# keypoints1 = [(x, y, size, angle, response, octave, class_id) for x, y, size, angle, response, octave, class_id in keypoints1]
# keypoints2 = [(x, y, size, angle, response, octave, class_id) for x, y, size, angle, response, octave, class_id in keypoints2]



# # print the length of the descriptors
# print(len(descriptors1))
# print(len(descriptors2))

# # Match the keypoints
# start_time = time.time()

# matches = brute_force_matcher(descriptors1, descriptors2)
# # matches = knn_matcher(descriptors1, descriptors2)

# end_time = time.time()
# print('Elapsed time: %.2f seconds' % (end_time - start_time))
# print('%d matches' % len(matches))

# # good_matches = []
# # for match in matches:
# #     if len(match) == 2:
# #             good_matches.append(match[0])

# # sort the matches
# matches = sorted(matches, key=lambda match: match.distance)
# # finding good matches
# new_matches = []
# # remove the matches such that the angle wrt x axis is greater than 30 degrees
# for match in matches:
#     x1, y1 = keypoints1[match.queryIdx][:2]
#     x2, y2 = keypoints2[match.trainIdx][:2]
#     angle = np.arctan((y2 - y1) / (x2 - x1))
#     angle = np.degrees(angle)
#     if abs(angle) < 20:
#         new_matches.append(match)

# # sort the matches
# new_matches = sorted(new_matches, key=lambda match: match.distance)

# good_matches = new_matches[:min(1000, len(new_matches))]





# # # Display the matches without using OpenCV
# # image_matches = np.concatenate((image1, image2), axis=1)
# # for match in  good_matches:
# #     image_matches = cv2.line(image_matches, (int(keypoints1[match[0]].pt[0]), int(keypoints1[match[0]].pt[1])), (int(keypoints2[match[1]].pt[0]) + image1.shape[1], int(keypoints2[match[1]].pt[1])), (0, 255, 0))
# # plt.imshow(image_matches)
# # plt.axis('off')

# #print the length of the good matches
# print(len(good_matches))

# # Draw the matches
# # convert matches to DMatch objects
# matches = []
# for match in good_matches:

#     matches.append(cv2.DMatch(match.queryIdx, match.trainIdx, match.distance))
# # convert the keypoints to cv2 keypoints
# keypoints1 = [cv2.KeyPoint(x, y, size, angle, response,int (octave), int(class_id)) for x, y, size, angle, response, octave, class_id in keypoints1]
# keypoints2 = [cv2.KeyPoint(x, y, size, angle, response, int(octave), int(class_id)) for x, y, size, angle, response, octave, class_id in keypoints2]
# image_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow('Matches', image_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# # estimate the homography
# src_points = np.array([keypoints1[match.queryIdx].pt for match in good_matches])
# dst_points = np.array([keypoints2[match.trainIdx].pt for match in good_matches])

# H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

# correspondences = []
# for match in good_matches:
#     x1, y1 = keypoints1[match.queryIdx].pt
#     x2, y2 = keypoints2[match.trainIdx].pt
#     correspondences.append([x1, y1, x2, y2])

# correspondences = np.matrix(correspondences)
# # run ransac
# H_custom, inliers = ransac(correspondences, 5)

# # compare the homography
# print(H)
# # print type 
# print("Type of opencv H: ", type(H))
# print(H_custom)
# print("Type of custom H: ", type(H_custom))
# # Difference
# print("Difference: ", np.linalg.norm(H - H_custom))




# # warp the images
# output_shape = (image1.shape[1]+image2.shape[1], image1.shape[0]+image2.shape[0])
# output_img = cv2.warpPerspective(image1, H_custom, output_shape)
# cv2.imshow('Warped Image with opencv', output_img)
# cv2.waitKey(0)

# # warp the images
# output_shape = (image1.shape[1]+image2.shape[1], image1.shape[0]+image2.shape[0])
# output_img = warpPerspective(image1, H_custom, output_shape)
# cv2.imshow('Warped Image with Custom', output_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# # def transform(src_pts, H):
# #     # src = [src_pts 1]
# #     src = np.pad(src_pts, [(0, 0), (0, 1)], constant_values=1)
# #     # pts = H * src
# #     pts = np.dot(H, src.T).T
# #     # normalize and throw z=1
# #     pts = (pts / pts[:, 2].reshape(-1, 1))[:, 0:2]
# #     return pts
# # width, height = image1.shape[1]+image2.shape[1], image1.shape[0]+image2.shape[0]

# # idx_pts = np.mgrid[0:width, 0:height].reshape(2, -1).T
# # map_pts = transform(idx_pts, np.linalg.inv(H))
# # map_pts = map_pts.reshape(width, height, 2).astype(np.float32)
# # warped = cv2.remap(image1, map_pts, None, cv2.INTER_CUBIC).transpose(1, 0, 2)
# # cv2.imshow('Warped Image with Custom', warped)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# # # # Warp the images
# # # output_shape = (image1.shape[1]+image2.shape[1], image1.shape[0]+image2.shape[0])
# # # output_img = warpPerspective(image1, H, output_shape)
# # # cv2.imshow('Warped Image with Custom', output_img)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()

