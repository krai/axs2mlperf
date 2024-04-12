
import os
import cv2
import numpy as np
import json
import sys
import shutil

def generate_file_list(supported_extensions, calibration_dir=None, index_file=None, first_n=None, first_n_insert=None, images_directory=None,):
    original_file_list = os.listdir(images_directory)
    sorted_filenames = [filename for filename in sorted(original_file_list) if any(filename.lower().endswith(extension) for extension in supported_extensions) ]

    if index_file:
        index_file = os.path.join(calibration_dir, index_file)
        with open(index_file, 'r') as file_in:
            sorted_filenames = []
            for line in file_in:
                sorted_filenames.append((line.split())[0])
                sorted_filenames.sort()
            first_n_insert = f'{index_file}_'
    elif first_n:
        sorted_filenames = sorted_filenames[:first_n] #if first_n is not None else sorted_filenames
        assert len(sorted_filenames) == first_n

    return sorted_filenames

def norma_layout(image_data, data_type, data_layout, subtract_mean, given_channel_means, normalize_symmetric):
    image_data = np.asarray(image_data, dtype=data_type)
    if normalize_symmetric is not None:
        if normalize_symmetric:
            image_data = image_data/127.5 - 1.0
        else:
            input_data = input_data/255.0
    # Subtract mean value.
    if subtract_mean:
        if len(given_channel_means):
            image_data -= given_channel_means
        else:
            image_data -= np.mean(image_data)
    # NHWC -> NCHW.
    if data_layout == 'NCHW':
        image_data = image_data[:,:,0:3].transpose(2, 0, 1)
    return image_data

# Load and preprocess image
def load_image(image_path,            # Full path to processing image
               target_size,           # Desired size of resulting image
               crop_percentage = 87.5,# Crop to this percentage then scale to target size
               data_layout = 'NHWC',  # Data layout to store
               convert_to_bgr = False,# Swap image channel RGB -> BGR
               interpolation_method = cv2.INTER_LINEAR # Interpolation method.
               ):

    out_height = target_size
    out_width  = target_size

    def resize_with_aspectratio(img):
        height, width, _ = img.shape
        new_height = int(100. * out_height / crop_percentage)   # intermediate oversized image from which to crop
        new_width = int(100. * out_width / crop_percentage)     # ---------------------- ,, ---------------------
        if height > width:
            w = new_width
            h = int(new_height * height / width)
        else:
            h = new_height
            w = int(new_width * width / height)
        img = cv2.resize(img, (w, h), interpolation = interpolation_method)
        return img

    def center_crop(img):
        height, width, _ = img.shape
        left = int((width - out_width) / 2)
        right = int((width + out_width) / 2)
        top = int((height - out_height) / 2)
        bottom = int((height + out_height) / 2)
        img = img[top:bottom, left:right]
        return img


    img = cv2.imread(image_path)
    if len(img.shape) < 3 or img.shape[2] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mimic preprocessing steps from the official reference code.
    img = resize_with_aspectratio(img)
    img = center_crop(img)
    # Convert to BGR.
    if convert_to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def preprocess_files(selected_filenames, images_directory, destination_dir, crop_percentage, resolution, convert_to_bgr,
    data_type, data_layout, new_file_extension, normalayout, subtract_mean, given_channel_means, normalize_symmetric, 
    quantized, quant_scale, quant_offset, convert_to_unsigned, interpolation_method):
    "Go through the selected_filenames and preprocess all the files (optionally normalize and subtract mean)"
    output_filenames = []

    for current_idx in range(len(selected_filenames)):
        input_filename = selected_filenames[current_idx]

        full_input_path = os.path.join(images_directory, input_filename)

        image_data = load_image(image_path = full_input_path,
                              target_size = resolution,
                              crop_percentage = crop_percentage,
                              convert_to_bgr = convert_to_bgr,
                              interpolation_method = interpolation_method)

        if quantized:
            image_data = norma_layout(image_data, data_type, data_layout, subtract_mean, given_channel_means, normalize_symmetric)
            image_data = quantized_to_int8(image_data, quant_scale, quant_offset)
            if convert_to_unsigned:
                image_data = int8_to_uint8(image_data)
        elif normalayout:
            image_data = norma_layout(image_data, data_type, data_layout, subtract_mean, given_channel_means, normalize_symmetric)

        output_filename = input_filename.rsplit('.', 1)[0] + '.' + new_file_extension if new_file_extension else input_filename

        output_filename_calib = input_filename.rsplit('.', 1)[0] + '.' + new_file_extension + '.raw'

        full_output_path = os.path.join(destination_dir, output_filename)
        image_data.tofile(full_output_path)

        print("[{}]:  Stored {}".format(current_idx+1, full_output_path) )

        output_filenames.append(output_filename)

    return output_filenames

def quantized_to_int8(image, scale, offset):
    quant_image = (image/scale + offset).astype(np.float32)
    output = np.copy(quant_image)
    gtZero = (quant_image > 0).astype(int)
    gtZero = gtZero * 0.5
    output=output+gtZero
    ltZero = (quant_image < 0).astype(int)
    ltZero = ltZero * (-0.5)
    output=output+ltZero
    return output.astype(np.int8)


def int8_to_uint8(image):
    image = (image+128).astype(np.uint8)
    return image

def preprocess(dataset_name, crop_percentage, resolution, convert_to_bgr,
    data_type, data_layout, new_file_extension, file_name, normalayout, subtract_mean, given_channel_means, 
    normalize_symmetric, quantized, quant_scale, offset, quant_offset, first_n, fof_name, image_file, convert_to_unsigned, 
    interpolation_method, supported_extensions, input_file_list, 
    index_file=None, tags=None, calibration=None, entry_name=None, images_directory=None, __record_entry__=None ):
    __record_entry__["tags"] = tags or [ "preprocessed" ]
    if quantized:
        entry_end = "_quantized"
    elif normalayout:
        entry_end = "_normalayout"
    else:
        entry_end = ""
    if not entry_name:
        if index_file:
            index_file_name = index_file.split('.')[0]
            first_n_insert = f'_{index_file_name}'
        else:
            first_n_insert = f'_first.{first_n}_images' if first_n and first_n != 50000 else ''

        entry_name = f'opencv_{dataset_name}_preprocessed_to.{resolution}{first_n_insert}{entry_end}'
    __record_entry__.save( entry_name )
    output_directory     = __record_entry__.get_path(file_name)

    os.makedirs( output_directory )
    destination_dir = output_directory

    print(("From: {}, To: {}, Size: {}, Crop: {}, 2BGR: {}, OFF: {}, VOL: '{}', FOF: {},"+
        " DTYPE: {}, DLAYOUT: {}, EXT: {}, NORM: {}, SMEAN: {}, GCM: {}, QUANTIZE: {}, QUANT_SCALE: {}, QUANT_OFFSET: {}, CONV_UNSIGNED: {}, INTER: {}, IMG: {}").format(
        images_directory, destination_dir, resolution, crop_percentage, convert_to_bgr, offset, first_n, fof_name,
        data_type, data_layout, new_file_extension, normalize_symmetric, subtract_mean, given_channel_means, quantized, quant_scale, quant_offset, convert_to_unsigned, interpolation_method, image_file) )

    if interpolation_method == 'INTER_AREA':

        # Used for ResNet in pre_process_vgg.
        interpolation_method = cv2.INTER_AREA
    else:
        # Default interpolation method.
        interpolation_method = cv2.INTER_LINEAR

    if image_file:
        
        images_directory          = os.path.dirname(image_file)
        selected_filenames  = [ os.path.basename(image_file) ]

    elif os.path.isdir(images_directory):
        total_volume = len(input_file_list)

        if index_file:
            selected_filenames = input_file_list
        else:
            selected_filenames = input_file_list[offset:offset+first_n]

    output_filenames = preprocess_files(
        selected_filenames, images_directory, destination_dir, crop_percentage, resolution, convert_to_bgr,
        data_type, data_layout, new_file_extension, normalayout, subtract_mean, given_channel_means, normalize_symmetric, quantized, quant_scale, quant_offset, convert_to_unsigned, interpolation_method)

    fof_full_path = os.path.join(destination_dir, fof_name)
    with open(fof_full_path, 'w') as fof:
        for filename in output_filenames:
            fof.write(filename + '\n')

    return __record_entry__
