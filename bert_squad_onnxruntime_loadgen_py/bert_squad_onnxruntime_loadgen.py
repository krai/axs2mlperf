#!/usr/bin/env python3

import json
import math
import os
import pickle
import subprocess
import sys
import numpy as np
import onnxruntime
import time
import array

import mlperf_loadgen as lg

input_parameters_file_path = sys.argv[1]
user_conf_path = sys.argv[2]

input_parameters = {}

with open(input_parameters_file_path) as f:
    input_parameters = json.load(f)

print("DEBUG: input_parameters = ", input_parameters)

bert_code_root = os.path.join( input_parameters["mlperf_inference_path"], 'language', 'bert')

sys.path.insert(0, bert_code_root)
sys.path.insert(0, os.path.join(bert_code_root,'DeepLearningExamples','TensorFlow','LanguageModeling','BERT'))

## SQuAD dataset - original and tokenized
#
squad_dataset_tokenized_path= input_parameters["tokenized_squad_path"]

## BERT model:
#
model_name                  = input_parameters["model_name"]
bert_model_path             = input_parameters["model_path"]
model_input_layers_tms      = eval(input_parameters["model_input_layers_tms"])

## Processing by batches:
#
batch_size                  = input_parameters["batch_size"]
execution_device            = input_parameters["execution_device"]

scenario_str                = input_parameters["loadgen_scenario"]
mode_str                    = input_parameters["loadgen_mode"]
dataset_size                = input_parameters["loadgen_dataset_size"]
buffer_size                 = input_parameters["loadgen_buffer_size"]
count_override              = input_parameters["loadgen_count_override"]
mlperf_conf_path            = input_parameters["loadgen_mlperf_conf_path"]
verbosity                   = input_parameters["verbosity"]

sess_options = onnxruntime.SessionOptions()

if execution_device == "cpu":
    requested_provider = "CPUExecutionProvider"
elif execution_device in ["gpu", "cuda"]:
    requested_provider = "CUDAExecutionProvider"

print("Loading BERT model and weights from {} ...".format(bert_model_path))
sess = onnxruntime.InferenceSession(bert_model_path, sess_options, providers= [requested_provider] if execution_device else onnxruntime.get_available_providers())

session_execution_provider=sess.get_providers()
print("Session execution provider: ", sess.get_providers(), file=sys.stderr)

if "CUDAExecutionProvider" in session_execution_provider or "TensorrtExecutionProvider" in session_execution_provider:
    print("Device: GPU", file=sys.stderr)
else:
    print("Device: CPU", file=sys.stderr)


def tick(letter, quantity=1):
    if verbosity:
        print(letter + (str(quantity) if quantity>1 else ''), end='')

def load_query_samples(sample_indices):
    if verbosity > 1:
        print("load_query_samples({})".format(sample_indices))

    len_sample_indices = len(sample_indices)
    tick('B', len_sample_indices)


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
        batch               = query_samples[j:j+batch_size]

        input_ids_stack     = []
        input_mask_stack    = []
        segment_ids_stack   = []

        for index_in_batch, qs in enumerate(batch):
            global_index = qs.index

            selected_feature = all_samples[global_index]

            input_ids_stack.append( np.array(selected_feature.input_ids) )
            input_mask_stack.append( np.array(selected_feature.input_mask) )
            segment_ids_stack.append( np.array(selected_feature.segment_ids) )

        input_dict = dict(zip(
            model_input_layers_tms,
            [
                np.stack( input_ids_stack ).astype(np.int64),
                np.stack( input_mask_stack ).astype(np.int64),
                np.stack( segment_ids_stack ).astype(np.int64),
            ]
        ))

        scores = sess.run([o.name for o in sess.get_outputs()], input_dict)
        output_logits = np.stack(scores, axis=-1)

        response            = []
        response_array_refs = []

        for index_in_batch, qs in enumerate(batch):
            response_array = array.array("B", np.array(output_logits[index_in_batch].tolist(), np.float32).tobytes())
            response_array_refs.append(response_array)
            bi = response_array.buffer_info()
            response.append(lg.QuerySampleResponse(qs.id, bi[0], bi[1]))
        lg.QuerySamplesComplete(response)
    sys.stdout.flush()

def flush_queries():
    pass

def benchmark_using_loadgen():
    "Perform the benchmark using python API for the LoadGen library"

    global all_samples

    print("Loading tokenized SQuAD dataset as features from {} ...".format(squad_dataset_tokenized_path))

    with open(squad_dataset_tokenized_path, 'rb') as tokenized_features_file:

        all_samples  = pickle.load(tokenized_features_file)

    print("Example width: {}".format(len(all_samples[0].input_ids)))

    total_examples  = len(all_samples)
    print("Total examples available: {}".format(total_examples))

    loadgen_dataset_size = dataset_size or total_examples
    print("Number of selected samples: {}".format(loadgen_dataset_size))

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
