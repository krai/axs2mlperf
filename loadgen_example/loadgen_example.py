#!/usr/bin/env python3

import array
import json
import random
import sys
import time

import numpy as np
import mlperf_loadgen as lg

input_parameters_file_path = sys.argv[1]
input_parameters = {}

with open(input_parameters_file_path) as f:
    input_parameters = json.load(f)

scenario_str                = input_parameters["loadgen_scenario"]
mode_str                    = input_parameters["loadgen_mode"]
dataset_size                = int( input_parameters["loadgen_dataset_size"] )
buffer_size                 = int( input_parameters["loadgen_buffer_size"] )
mlperf_conf_path            = input_parameters["loadgen_mlperf_conf_path"]
user_conf_path              = input_parameters["loadgen_user_conf_path"]
model_name                  = input_parameters["model_name"]
latency_ms                  = int( input_parameters["latency_ms"] )

dataset         = [10*i for i in range(dataset_size)]
labelset        = [10*i+random.randint(0,1) for i in range(dataset_size)]


def predict_label(x_vector):
    time.sleep(latency_ms/1000.0)   # fractional seconds
    return int(x_vector/10)+1


def issue_queries(query_samples):

    printable_query = [(qs.index, qs.id) for qs in query_samples]
    print("LG: issue_queries( {} )".format(printable_query))

    predicted_results = {}
    for qs in query_samples:
        query_index, query_id = qs.index, qs.id

        x_vector        = dataset[query_index]
        predicted_label = predict_label(x_vector)

        predicted_results[query_index] = predicted_label
    print("LG: predicted_results = {}".format(predicted_results))

    response = []
    for qs in query_samples:
        query_index, query_id = qs.index, qs.id

        response_array = array.array("B", np.array(predicted_results[query_index], np.float32).tobytes())
        bi = response_array.buffer_info()
        response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
    lg.QuerySamplesComplete(response)


def flush_queries():
    print("LG called flush_queries()")


def load_query_samples(sample_indices):
    print("LG called load_query_samples({})".format(sample_indices))


def unload_query_samples(sample_indices):
    print("LG called unload_query_samples({})".format(sample_indices))
    print("")


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

