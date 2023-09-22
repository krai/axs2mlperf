#!/usr/bin/env python3

"""Loadgen-based Image Classification program based on ONNX.

Usage examples  :

                    # full Accuracy run (query mode) :
                axs byquery loadgen_output,task=image_classification,framework=onnxrt,preprocessed_imagenet_dir=/datasets/imagenet/pillow_sq.224_cropped_resized_imagenet50000,loadgen_scenario=Offline,loadgen_dataset_size=50000,loadgen_buffer_size=1000,batch_size=1000 , get accuracy
"""

import array
import os
import sys
import time
import json

import numpy as np
import onnxruntime as rt
from imagenet_loader import ImagenetLoader

import mlperf_loadgen as lg

input_parameters_file_path = sys.argv[1]

data_layout                 = 'NCHW'

MODEL_IMAGE_CHANNELS        = 3
MODEL_IMAGE_HEIGHT          = 224
MODEL_IMAGE_WIDTH           = 224
MODEL_INPUT_DATA_TYPE       = 'float32'

input_parameters = {}

with open(input_parameters_file_path) as f:
    input_parameters = json.load(f)

scenario_str                = input_parameters["loadgen_scenario"]
mode_str                    = input_parameters["loadgen_mode"]
dataset_size                = input_parameters["loadgen_dataset_size"]
buffer_size                 = input_parameters["loadgen_buffer_size"]
count_override              = input_parameters["loadgen_count_override"]
multistreamness             = input_parameters["loadgen_multistreamness"]
mlperf_conf_path            = input_parameters["loadgen_mlperf_conf_path"]
user_conf_path              = input_parameters["loadgen_user_conf_path"]
verbosity                   = input_parameters["verbosity"]
model_path                  = input_parameters["model_path"]
model_name                  = input_parameters["model_name"]
normalize_symmetric         = eval(input_parameters["normalize_symmetric"])
subtract_mean_bool          = eval(input_parameters["subtract_mean_bool"])
given_channel_means         = eval(input_parameters["given_channel_means"])
output_layer_name           = input_parameters["output_layer_name"]
execution_device            = input_parameters["execution_device"]         # if empty, it will be autodetected
batch_size                  = input_parameters["batch_size"]
cpu_threads                 = input_parameters["cpu_threads"]
preprocessed_imagenet_dir   = input_parameters["preprocessed_images_dir"]
input_file_list             = input_parameters["input_file_list"]

given_channel_stds          = []

loader_object               = ImagenetLoader(preprocessed_imagenet_dir, input_file_list, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH, data_layout, normalize_symmetric, subtract_mean_bool, given_channel_means, given_channel_stds)
preprocessed_image_buffer   = None
preprocessed_image_map      = np.empty(dataset_size, dtype=np.int32)   # this type should be able to hold indices in range 0:dataset_size


sess_options = rt.SessionOptions()
if cpu_threads > 0:
    sess_options.enable_sequential_execution = False
    sess_options.session_thread_pool_size = cpu_threads

if execution_device == "cpu":
    requested_provider = "CPUExecutionProvider"
elif execution_device in ["gpu", "cuda"]:
    requested_provider = "CUDAExecutionProvider"
elif execution_device in ["tensorrt", "trt"]:
    requested_provider = "TensorrtExecutionProvider"

sess = rt.InferenceSession(model_path, sess_options, providers= [requested_provider] if execution_device else rt.get_available_providers())

session_execution_provider=sess.get_providers()
print("Session execution provider: ", sess.get_providers(), file=sys.stderr)

if "CUDAExecutionProvider" in session_execution_provider or "TensorrtExecutionProvider" in session_execution_provider:
    print("Device: GPU", file=sys.stderr)
else:
    print("Device: CPU", file=sys.stderr)

input_layer_names   = [ x.name for x in sess.get_inputs() ]
input_layer_name    = input_layer_names[0]

if output_layer_name:
    for i, output_layer in enumerate(sess.get_outputs()):
        if output_layer.name == output_layer_name:
            output_layer_index = i
            break
else:
    output_layer_index  = 0
    output_layer_name   = sess.get_outputs()[output_layer_index].name

model_input_shape   = sess.get_inputs()[0].shape
height              = model_input_shape[2]
width               = model_input_shape[3]
model_output_shape  = sess.get_outputs()[output_layer_index].shape


def tick(letter, quantity=1):
    if verbosity:
        print(letter + (str(quantity) if quantity>1 else ''), end='')

def load_query_samples(sample_indices):     # 0-based indices in our whole dataset
    global preprocessed_image_buffer

    if verbosity > 1:
        print("load_query_samples({})".format(sample_indices))

    len_sample_indices = len(sample_indices)

    tick('B', len_sample_indices)

    if preprocessed_image_buffer is None:     # only do this once, once we know the expected size of the buffer
        preprocessed_image_buffer = np.empty((len_sample_indices, MODEL_IMAGE_CHANNELS, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_WIDTH), dtype=MODEL_INPUT_DATA_TYPE)

    for buffer_index, sample_index in zip(range(len_sample_indices), sample_indices):
        preprocessed_image_map[sample_index] = buffer_index
        preprocessed_image_buffer[buffer_index] = np.array( loader_object.load_image_by_index_and_normalize(sample_index)[0] )

        tick('l')

    if verbosity:
        print('')

def unload_query_samples(sample_indices):
    #print("unload_query_samples({})".format(sample_indices))
    tick('U')

    if verbosity:
        print('')

def issue_queries(query_samples):

    if verbosity > 2:
        printable_query = [(qs.index, qs.id) for qs in query_samples]
        print("issue_queries( {} )".format(printable_query))
    tick('Q', len(query_samples))

    for j in range(0, len(query_samples), batch_size):
        batch       = query_samples[j:j+batch_size]   # NB: the last one may be shorter than batch_size in length
        batch_data  = preprocessed_image_buffer[preprocessed_image_map[ [qs.index for qs in batch] ]]

        begin_time = time.time()

        batch_predictions   = sess.run([output_layer_name], {input_layer_name: batch_data})[0]

        inference_time_s = time.time() - begin_time

#        actual_batch_size = len(batch_predictions)
        actual_batch_size = "(unknown)"

        if verbosity > 1:
            print("[batch of {}] inference={:.2f} ms".format(len(batch), inference_time_s*1000))

        batch_predicted_labels  = (np.argmax( batch_predictions, axis=1) - 1).tolist()

        tick('p', len(batch))
        if verbosity > 2:
            print("predicted_batch_results = {}".format(batch_predicted_labels))

        response = []
        response_array_refs = []    # This is needed to guarantee that the individual buffers to which we keep extra-Pythonian references, do not get garbage-collected.
        for qs, predicted_label in zip(batch, batch_predicted_labels):

            response_array = array.array("B", np.array(predicted_label, np.float32).tobytes())
            response_array_refs.append(response_array)
            bi = response_array.buffer_info()
            response.append(lg.QuerySampleResponse(qs.id, bi[0], bi[1]))
        lg.QuerySamplesComplete(response)
        #tick('R', len(response))
    sys.stdout.flush()

def flush_queries():
    pass

def benchmark_using_loadgen():
    "Perform the benchmark using python API for the LoadGen library"

    scenario = {
        'SingleStream':     lg.TestScenario.SingleStream,
        'MultiStream':      lg.TestScenario.MultiStream,
        'Server':           lg.TestScenario.Server,
        'Offline':          lg.TestScenario.Offline,
    }[scenario_str]

    mode = {
        'AccuracyOnly':     lg.TestMode.AccuracyOnly,
        'PerformanceOnly':  lg.TestMode.PerformanceOnly,
        'SubmissionRun':    lg.TestMode.SubmissionRun,
    }[mode_str]

    ts = lg.TestSettings()
    if(mlperf_conf_path):
        ts.FromConfig(mlperf_conf_path, model_name, scenario_str)
    if(user_conf_path):
        ts.FromConfig(user_conf_path, model_name, scenario_str)
    ts.scenario = scenario
    ts.mode     = mode

    if multistreamness is not None:
        ts.multi_stream_samples_per_query = multistreamness
    if count_override is not None:
        ts.min_query_count = count_override
        ts.max_query_count = count_override

    sut = lg.ConstructSUT(issue_queries, flush_queries)
    qsl = lg.ConstructQSL(dataset_size, buffer_size, load_query_samples, unload_query_samples)

    log_settings = lg.LogSettings()
    log_settings.enable_trace = False
    lg.StartTestWithLogSettings(sut, qsl, ts, log_settings)

    lg.DestroyQSL(qsl)
    lg.DestroySUT(sut)


try:
    benchmark_using_loadgen()
except Exception as e:
    print('{}'.format(e))

