# coding: utf-8
import os
from os import listdir
from os.path import isfile, join
import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import constant


from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import time
import OpenEXR, array

np.set_printoptions(threshold=sys.maxsize, suppress=True, linewidth=2000)
refit_response_curve = False
if refit_response_curve:
    disable_eager_execution()


import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

dont_use_gpu=True
if dont_use_gpu: # GPU with 4GM memory runs out of memory if there are more than 6 images
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    try:
        # Disable all GPUS
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass



def pngs_to_exr_to_png(project_name="", source_directory="/home/sense/3cobot/", refit_response_curve=False, photo_filenames=None, exposure_times=None, save_exr=False):
    start_time = time.time()
    radiances = make_hdr(project_name=project_name, source_directory=source_directory, start_time=start_time, refit_response_curve=refit_response_curve, start_index=0, end_index=-1, squeeze_files_by=1, number_of_samples_per_dimension=50, photo_filenames=photo_filenames, exposure_times=exposure_times)
    #save_radiance_map(project_name=project_name, radiances=radiances,)
    save_exr_file(radiances, project_name)
    tonemapped_png, raw_saturation_corrected_hdr_data = exr_to_png(project_name)
    if not save_exr:
        exr_filename = "{}.exr".format(project_name)
        os.remove(exr_filename)

    return tonemapped_png, raw_saturation_corrected_hdr_data

def save_radiance_map(project_name, radiances):
    #print('Saving radiance map')
    plt.figure(figsize=(20,20))
    plt.imshow(np.log2(cv2.cvtColor(radiances, cv2.COLOR_RGB2GRAY)), cmap='jet')
    plt.colorbar()
    plt.savefig('{}_radiances.png'.format(project_name))

def load_images_and_exposure_times(project_name, source_directory, squeeze_files_by=1, start_index=0, end_index=-1, crop_x_min=0, crop_y_min=0, crop_x_max=2048, crop_y_max=2048, photo_filenames=None, exposure_times=None):
    if photo_filenames and exposure_times:
        files_in_directory = photo_filenames #["{}{}".format(source_directory, photo_filename) for photo_filename in photo_filenames]
    else:
        files_in_directory = [f for f in listdir(source_directory) if isfile(join(source_directory, f)) and "{}".format(project_name) in f and ".png" in f and "photo" in f]
        exposure_times = [float(file_name.split("{}_".format(project_name))[1].split("ms")[0]) for file_name in files_in_directory] 
        print(exposure_times)
        exposure_times.sort()
        files_in_directory = ["{}_{:.3f}ms.png".format(project_name, exposure_time) for exposure_time in exposure_times]

    exposure_times = [e/1000.0 for e in exposure_times]
    files_in_directory = files_in_directory[start_index::squeeze_files_by]
    exposure_times = exposure_times[start_index::squeeze_files_by]
    #print("Working with {} exposure times: {} seconds".format(len(exposure_times), exposure_times))
    #print("...corresponding to these files: {}".format(files_in_directory))
    images = [cv2.imread( "{}/{}".format(source_directory, file_name) , 1) for file_name in files_in_directory]
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    images = np.asarray(images)
    #print("Shape of images before crop: {}".format(images.shape))
    try:
        images = images[:,crop_y_min:crop_y_max, crop_x_min:crop_x_max, :]
        #print("Shape of images after crop by height: [{}: {}], width: [{}:{}] = {}".format(crop_x_min, crop_x_max, crop_y_min, crop_y_max, images.shape))
        return images, exposure_times
    except IndexError:
        print("No HDR images to load, moving on")
        sys.exit(0)


def hdr_debvec(img_list, exposure_times, number_of_samples_per_dimension):
    B = [math.log(e,2) for e in exposure_times]
    l = constant.L
    w = [z if z <= 0.5*255 else 255-z for z in range(256)]

    width = img_list[0].shape[0]
    height = img_list[0].shape[1]
    width_iteration = math.floor(width / number_of_samples_per_dimension)
    height_iteration = math.floor(height / number_of_samples_per_dimension)
    #print("Striding image by {} x {}".format(width_iteration, height_iteration))

    w_iter = 0
    h_iter = 0

    Z = np.zeros((len(img_list), number_of_samples_per_dimension*number_of_samples_per_dimension))
    for img_index, img in enumerate(img_list):
        indexed_item = img[0:width:width_iteration, 0:height:height_iteration].flatten()
        indexed_item = indexed_item[:Z.shape[1]]
        Z[img_index, :] = indexed_item

   # print("Finished sampling from image...")
    return response_curve_solver(Z, B, l, w)

def response_curve_solver(Z, B, l, w):
    n = 256
    A = np.zeros(shape=(np.size(Z, 0)*np.size(Z, 1)+n+1, n+np.size(Z, 1)), dtype=np.float32)
    b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)

    k = 0
    for i in range(np.size(Z, 1)):
        for j in range(np.size(Z, 0)):
            z = int(Z[j][i])
            wij = w[z]
            A[k][z] = wij
            A[k][n+i] = -wij
            b[k] = wij*B[j]
            k += 1
    
    # Fix the curve by setting its middle value to 0
    A[k][128] = 1
    k += 1

    # Include the smoothness equations
    for i in range(n-1):
        A[k][i]   =    l*w[i+1]
        A[k][i+1] = -2*l*w[i+1]
        A[k][i+2] =    l*w[i+1]
        k += 1

    # Solve the system using SVD
    x = np.linalg.lstsq(A, b)[0]
    g = x[:256]
    lE = x[256:]

    return g, lE


def make_hdr(project_name, source_directory, start_time, refit_response_curve=False, start_index=0, end_index=-1, squeeze_files_by=1, number_of_samples_per_dimension=50, photo_filenames=None, exposure_times=None):
    start_load_time = time.time()
    #print("Beginning to start loading of images at {} seconds".format(start_load_time - start_time))

    images, exposure_times = load_images_and_exposure_times(project_name=project_name, source_directory=source_directory, start_index=start_index, end_index=end_index, squeeze_files_by=squeeze_files_by, photo_filenames=photo_filenames, exposure_times=exposure_times)

    load_time = time.time()
    #print("{} images loaded by OpenCV at {} seconds".format(images.shape[0], load_time - start_time))

    if refit_response_curve:
        images_red_channel = images[:,:,:,0]
       	images_green_channel = images[:,:,:,1]
        images_blue_channel = images[:,:,:,2]

        #print('Solving response curve for green .... ', flush=True)
        gg, _ = hdr_debvec(images_green_channel, exposure_times, number_of_samples_per_dimension=number_of_samples_per_dimension)

        #print('Solving response curve for red .... ', flush=True)
        gr, _ = hdr_debvec(images_red_channel, exposure_times, number_of_samples_per_dimension=number_of_samples_per_dimension)

        #print('Solving response curve for blue .... ', flush=True)
        gb, _ = hdr_debvec(images_blue_channel, exposure_times, number_of_samples_per_dimension=number_of_samples_per_dimension)
        # print('done')

        with open('camera_response_function_red.npy', 'wb') as f:
            np.save(f, gr)

        with open('camera_response_function_green.npy', 'wb') as f:
            np.save(f, gg)

        with open('camera_response_function_blue.npy', 'wb') as f:
            np.save(f, gb)

        # Show response curve
        #print('Saving response curves plot .... ', end='')
        plt.figure(figsize=(10, 10))
        plt.plot(gr, range(256), 'rx')
        plt.plot(gg, range(256), 'gx')
        plt.plot(gb, range(256), 'bx')
        plt.ylabel('pixel value Z')
        plt.xlabel('log exposure X')
        plt.savefig('response-curves.png')

        curves_solved = time.time()
        #print("RGB response curves saved to response-curves.png, solved at {} seconds".format(curves_solved - start_time))

        sys.exit(0)

    red_pixel_intensity_map = np.load('camera_response_function_red.npy')
    green_pixel_intensity_map = np.load('camera_response_function_green.npy')
    blue_pixel_intensity_map = np.load('camera_response_function_blue.npy')
    pixel_intensity_map = np.asarray([red_pixel_intensity_map, green_pixel_intensity_map, blue_pixel_intensity_map])

    next_load_time = time.time()
    #print("Camera response data loaded at {} seconds".format(next_load_time - start_time))

    weights = np.asarray([pixel_itensity if pixel_itensity <= 0.5*255 else 255-pixel_itensity for pixel_itensity in range(256)])
    log_exposures = np.log2(exposure_times)

    load_time_3 = time.time()
    #print("NumPy quick computations of weights and log exposures at {} seconds".format(load_time_3 - start_time))

    number_of_images, image_width, image_height, color_channels = images.shape

    pixel_intensity_map_tensor = tf.squeeze(tf.convert_to_tensor(pixel_intensity_map, dtype=tf.float32))
    images_tensor = tf.convert_to_tensor(images, dtype=tf.int32)
    log_exposures_tensor = tf.convert_to_tensor(log_exposures, dtype=tf.float32)
    weights_tensor = tf.convert_to_tensor(weights, dtype=tf.float32)

    load_time_4 = time.time()
    #print("Conversions to tensors finished at {} seconds".format(load_time_4 - start_time))

    r_radiances = tf.gather(params=pixel_intensity_map_tensor[1], indices=images_tensor[:,:,:,0]) # hack: set R response to B response, until fix of R curve
    g_radiances = tf.gather(params=pixel_intensity_map_tensor[1], indices=images_tensor[:,:,:,1])
    b_radiances = tf.gather(params=pixel_intensity_map_tensor[2], indices=images_tensor[:,:,:,2])

    load_time_5 = time.time()
    #print("Mapping images to pixel intensities finished at {} seconds".format(load_time_5 - start_time))

    radiances = tf.convert_to_tensor([r_radiances,g_radiances,b_radiances])
    radiances = tf.transpose(a=radiances, perm=(2, 3, 0, 1))
    image_weights = tf.gather(params=weights_tensor, indices=images_tensor)
    image_weights = tf.transpose(a=image_weights, perm=(1, 2, 3, 0))

    load_time_6 = time.time()
    #print("Further gather and transposes finished at {} seconds".format(load_time_6 - start_time))

    broadcasted_log_exposures = log_exposures_tensor * tf.ones((image_width, image_height, color_channels, number_of_images))
    radiance_differences = radiances - broadcasted_log_exposures
    normalized_radiance_differences = image_weights * radiance_differences
 
    load_time_7 = time.time()
    #print("Broadcasting and multiplications finished at {} seconds".format(load_time_7 - start_time))

    radiance_accumulation = tf.math.reduce_sum(normalized_radiance_differences, axis=-1).numpy()
    weights_accumulation = tf.math.reduce_sum(image_weights, axis=-1).numpy()

    load_time_8 = time.time()
    #print("Summations finished at {} seconds".format(load_time_8 - start_time))

    division_result = np.divide(radiance_accumulation, weights_accumulation, out = np.zeros_like(radiance_accumulation), where = weights_accumulation != 0)
    normalized_radiance = np.exp(division_result)

    load_time_9 = time.time()
    #print("NumPy divide and exponentiation finished at {} seconds".format(load_time_9 - start_time))

    return normalized_radiance


def save_exr_file(radiances, project_name):
    width = radiances.shape[0]
    height = radiances.shape[1]

    # print("Average red value: {}".format(np.average(radiances[:,:,0])))
    # print("Average green value: {}".format(np.average(radiances[:,:,1])))
    # print("Average blue value: {}".format(np.average(radiances[:,:,2])))

    # print("Min red value: {}".format(np.amin(radiances[:,:,0])))
    # print("Min green value: {}".format(np.amin(radiances[:,:,1])))
    # print("Min blue value: {}".format(np.amin(radiances[:,:,2])))

    # print("Max red value: {}".format(np.amax(radiances[:,:,0])))
    # print("Max green value: {}".format(np.amax(radiances[:,:,1])))
    # print("Max blue value: {}".format(np.amax(radiances[:,:,2])))

    reduction_multiplier = 1 #2^12
    header = OpenEXR.Header(height, width)
    #print(header)

    r_data = array.array('f', radiances[:,:,0].flatten() / reduction_multiplier ).tostring()
    g_data = array.array('f', radiances[:,:,1].flatten() / reduction_multiplier ).tostring()
    b_data = array.array('f', radiances[:,:,2].flatten() / reduction_multiplier ).tostring()

    exr_filename = "{}.exr".format(project_name)
    exr = OpenEXR.OutputFile(exr_filename, header)
    exr.writePixels({'R': r_data, 'G': g_data, 'B': b_data})
    exr.close()
    return exr_filename

def convert_linear_to_srgb(color, defog_for_color, im, brilliance_to_balance=0.0):

    def knee(x, f):
        return np.log(x * f + 1) / f

    def find_knee_f(x, y):
        f0 = 0
        f1 = 1
        while knee(x, f1) > y:
            f0 = f1
            f1 = f1 * 2
        for i in range(0,30):
            f2 = (f0 + f1) / 2.0
            y2 = knee(x, f2)
            if y2 < y:
                f1 = f2
            else:
                f0 = f2
        return (f0 + f1) / 2.0

    exposure = 0.0
    exposure_factor = 2.47393
    knee_low = 0
    knee_high = 15.0 + brilliance_to_balance

    d = 0.001 * defog_for_color
    #print("Defog for {}: {} ({} after scaling)".format(color, defog_for_color, d))

    g = 2.2
    k1 = np.power(2, knee_low)
    x = np.power(2, knee_high) - k1
    y = np.power(2, 3.5) - k1
    m = np.power(2, exposure + exposure_factor)

    im = im - d # defog
    stuff = im < 0
    im[im < 0] = 0 # clip negative values to zero
    im = im * m # exposure
    f =  find_knee_f( np.power(2, knee_high) - k1, np.power(2, 3.5) - k1)
    true_change_value = k1 + knee(im - k1, f)

    im = np.where(im > k1, true_change_value, im)
    im = np.power(im, g)

    scaling_factor = np.power(2, -3.5 * g)
    #print("Scaling factor: {}".format(scaling_factor))
    im = im * 255.0 * scaling_factor
    im = np.clip(im, a_min = 0, a_max = 255)


    return im

def auto_adjustments_with_convert_scale_abs(img):
    alow = img.min()
    ahigh = img.max()
    amax = 255
    amin = 0

    # calculate alpha, beta
    alpha = ((amax - amin) / (ahigh - alow))
    beta = amin - alow * alpha
    # perform the operation g(x,y)= α * f(x,y)+ β
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return new_img

def exr_to_png(project_name):
    exr_image_to_read = "{}.exr".format(project_name)

    im = cv2.imread(exr_image_to_read,-1) 

    # where any one color channel is under-saturated (value of 1.0), recognize the aggregate color will also be partially saturated, so clamp the other channels to be 1.0, and thus a saturated black color (easy to filter)
    im[:,:,0][im[:,:,1] == 1.0] = 1.0
    im[:,:,0][im[:,:,2] == 1.0] = 1.0
    im[:,:,1][im[:,:,0] == 1.0] = 1.0
    im[:,:,1][im[:,:,2] == 1.0] = 1.0
    im[:,:,2][im[:,:,0] == 1.0] = 1.0
    im[:,:,2][im[:,:,1] == 1.0] = 1.0

    raw_saturation_corrected_hdr_data = im

    multiplier = 2*12
    im = im * multiplier

    image_width = im.shape[0]
    image_height = im.shape[1]
    image_pixels = image_width * image_height
    # print("Reading {} image {} x {} pixels".format(exr_image_to_read, image_width, image_height))


    r_fog_color =  np.sum(im[:,:,0]) / image_pixels 
    g_fog_color =  np.sum(im[:,:,1]) / image_pixels
    b_fog_color =  np.sum(im[:,:,2]) / image_pixels

    plus_minus_brilliance_to_balance = [-5.0, -2.5, 0.0, 2.5, 5.0]
    brilliance_descriptors = ["ultra-brilliant", "brilliant", "default", "flat", "ultra-flat"]
    output_filenames = []
    for brilliance_descriptor, brilliance_to_balance in zip(brilliance_descriptors, plus_minus_brilliance_to_balance):

        new_image = im.copy()
    
        new_image[:,:,0] = convert_linear_to_srgb("red", r_fog_color, new_image[:,:,0], brilliance_to_balance)
        new_image[:,:,1] = convert_linear_to_srgb("green", g_fog_color, new_image[:,:,1], brilliance_to_balance)
        new_image[:,:,2] = convert_linear_to_srgb("blue", b_fog_color, new_image[:,:,2], brilliance_to_balance)

        #raw_project_name = project_name.split("-")
        png_output_filename = "{}_color_balanced_from_hdr.png".format(project_name, brilliance_descriptor)

        if brilliance_descriptor == "flat":
            cv2.imwrite(png_output_filename,new_image)
            output_filenames.append(png_output_filename)

        new_image = auto_adjustments_with_convert_scale_abs(new_image)


        # img_yuv = cv2.cvtColor(new_image, cv2.COLOR_BGR2YUV)

        # # img_yuv = np.int16(img_yuv)
        # # img_yuv = np.clip(img_yuv, 0, 255)
        # # img_yuv = np.uint8(img_yuv)

        # y, u, v = cv2.split(img_yuv)
        # y = cv2.equalizeHist(y)
        # yuv = cv2.merge((y, u, v))

        #image_out = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


    return output_filenames[0], raw_saturation_corrected_hdr_data # currently ultra brilliant is default

if __name__ == "__main__":
    pngs_to_exr_to_png(project_name="crystal-1-", photo_filenames=["crystal-1-0_photo_0.png", "crystal-1-1_photo_0.png", "crystal-1-2_photo_0.png", "crystal-1-3_photo_0.png", "crystal-1-4_photo_0.png"], exposure_times=[25.0, 75.0, 150.0, 275.0, 400.0])