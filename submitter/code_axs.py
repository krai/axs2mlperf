#!/usr/bin/env python3


import json
import os
import shutil
import subprocess
import sys
from typing import ( Dict, List, Tuple )
from pathlib import Path
import random
from tqdm import tqdm

random.seed(0)

# Define the attributes of different program types in a constant dictionary
PROGRAM_ATTRIBUTES = {
    "image_classification_onnx_loadgen_py": {"tags": ["loadgen_output", "classified_imagenet"], "dataset_size": 50000, "buffer_size": 1024},
    "image_classification_torch_loadgen_py": {"tags": ["loadgen_output", "classified_imagenet"], "dataset_size": 50000, "buffer_size": 1024},
    "bert_squad_kilt_loadgen_qaic": {"tags": ["loadgen_output", "bert_squad", "device=qaic"], "dataset_size": 10833, "buffer_size": 10833},
    "bert_squad_kilt_loadgen_onnxrt": {"tags": ["loadgen_output", "bert_squad", "device=onnxrt"], "dataset_size": 10833, "buffer_size": 10833},
    "resnet50_kilt_loadgen_qaic": {"tags": ["loadgen_output", "image_classifier", "device=qaic"], "dataset_size": 50000, "buffer_size": 1024},
    "retinanet_kilt_loadgen_qaic": {"tags": ["loadgen_output", "object_detection", "device=qaic"], "dataset_size": 24781, "buffer_size": 64},
    # old names
    "object_detection_onnx_loadgen_py": {"tags": ["loadgen_output", "detected_coco"], "dataset_size": 24781, "buffer_size": 64, "retinanet_coco": {"dataset_size": 5000, "buffer_size": 64}},
    "bert_squad_onnxruntime_loadgen_py": {"tags": ["loadgen_output", "bert_squad"], "dataset_size": 10833, "buffer_size": 10833},
    "bert_squad_kilt_loadgen_c": {"tags": ["loadgen_output", "bert_squad", "device=qaic"], "dataset_size": 10833, "buffer_size": 10833},
}


SUPPORTED_MODELS = {"resnet50", "retinanet", "bert-99", "bert-99.9", "retinanet_coco"}
compliance_test_list = []

def store_json(data_structure, json_file_path):
    json_data   = json.dumps( data_structure , indent=4)

    with open(json_file_path, "w") as json_fd:
        json_fd.write( json_data+"\n" )

def get_mlperf_model_name(model_name_dict, model_name):
    if model_name in model_name_dict.keys():
        return model_name_dict[model_name]
    else:
        return None

def get_scenarios(sut_system_type, model_name):
    """
    Based on the system type and model name, this function decides which scenarios should be used.
    ----------------------------------------------
    edge: Offline, SingleStream, MultiStream
    datacenter: Offline, Server
    ----------------------------------------------
    Args:
    sut_system_type: The type of system under test
    model_name: The name of the model being used

    Returns:
    A list of scenarios
    """
    if sut_system_type == "edge":
        return ["Offline", "SingleStream", "MultiStream"] if model_name in ("resnet50", "retinanet") else ["Offline", "SingleStream"]
    elif sut_system_type == "datacenter":
        return ["Offline", "Server"]

def get_common_attributes_and_experiment_tags(program_name, framework, model_name, sut_name):
    """
    This function fetches common attributes and experiment tags based on the given program name.

    Args:
    program_name: The name of the program
    framework: The framework being used
    model_name: The name of the model
    sut_name: The name of the system under test

    Returns:
    A tuple containing a dictionary of common attributes and a list of experiment tags
    """
    # Modify the program_name if there's a mapping for the model_name
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model {model_name} is not supported")

    common_attributes = {"framework": framework, "model_name": model_name, "sut_name": sut_name}
    if program_name in PROGRAM_ATTRIBUTES:
        attributes = PROGRAM_ATTRIBUTES[program_name]
        if program_name == "object_detection_onnx_loadgen_py" and model_name == "retinanet_coco":
            attributes["dataset_size"] = attributes["retinanet_coco"]["dataset_size"]
            attributes["buffer_size"] = attributes["retinanet_coco"]["buffer_size"]
        common_attributes.update({
            # "loadgen_dataset_size": attributes["dataset_size"],
            # "loadgen_buffer_size": attributes["buffer_size"]
            "first_n": attributes["dataset_size"]
        })
        #TODO: so many conditons, remove first_n if it is possible in the future
        if program_name == "retinanet_kilt_loadgen_qaic":
            common_attributes["first_n"] = attributes["dataset_size"]
        return common_attributes, attributes["tags"]

def get_modes(division, model_name):
    """
    This function decides which modes should be used based on the division and the given program name.

    Args:
    division: The division of the experiment
    model_name: The name of the model

    Returns:
    A list of modes
    """
    modes = [
        ["loadgen_mode=AccuracyOnly"],
        ["loadgen_mode=PerformanceOnly", "loadgen_compliance_test-"],
    ]
    if division == "closed":
        compliance_tests_conditions = {
            "resnet50": ['TEST01', 'TEST04', 'TEST05'],
            "bert-99": ['TEST01', 'TEST05'],
            "bert-99.9": ['TEST01', 'TEST05'],
            "retinanet": ['TEST01', 'TEST05']
        }
        for name, tests in compliance_tests_conditions.items():
            if name in model_name:
                modes.extend([["loadgen_mode=PerformanceOnly", f"loadgen_compliance_test={test}"] for test in tests])
                break
    return modes

def get_scenario_attributes(scenario, mode_attribs, __entry__, loadgen_server_target_qps):
    """
    This function generates scenario attributes.

    Args:
    scenario: The current scenario
    mode_attribs: A list of mode attributes
    __entry__: The current experiment entry
    loadgen_server_target_qps: The target queries per second for the loadgen server

    Returns:
    A dictionary of scenario attributes
    """
    scenario_attributes = {"loadgen_scenario": scenario}

    if "loadgen_mode=AccuracyOnly" in mode_attribs and scenario == "Server":
        scenario_attributes["loadgen_target_qps"] = loadgen_server_target_qps if loadgen_server_target_qps is not None else __entry__["loadgen_target_qps"]

    elif "loadgen_mode=PerformanceOnly" in mode_attribs and scenario in ("Offline", "Server"):
        if scenario == "Server":
            scenario_attributes["loadgen_target_qps"] = loadgen_server_target_qps if loadgen_server_target_qps is not None else __entry__["loadgen_target_qps"]
        else:
            scenario_attributes["loadgen_target_qps"] = __entry__["loadgen_target_qps"]
            
        if scenario in ("SingleStream", "MultiStream"):
            scenario_attributes["loadgen_target_latency"] = __entry__["loadgen_target_latency"]
        elif scenario == "MultiStream":
            scenario_attributes["loadgen_multistreamness"] = __entry__["loadgen_multistreamness"]

    return scenario_attributes

def get_experiment_query(experiment_tags, common_attributes, mode_attribs, scenario_attributes):
    """
    This function generates the experiment query.

    Args:
    experiment_tags: A list of experiment tags
    common_attributes: A dictionary of common attributes
    mode_attribs: A list of mode attributes
    scenario_attributes: A dictionary of scenario attributes

    Returns:
    A string that is the joined query
    """
    # Convert dictionaries to list of strings
    common_attr_list = [f"{k}={common_attributes[k]}" for k in common_attributes]
    scenario_attr_list = [f"{k}={scenario_attributes[k]}" for k in scenario_attributes]

    # Create a combined list
    combined_list = experiment_tags + common_attr_list + mode_attribs + scenario_attr_list

    # Join the list into a string
    joined_query = ','.join(combined_list)
    
    return joined_query

def append_experiment_entries(scenario, mode_attribs, experiment_tags, common_attributes, __entry__, experiment_list_only, loadgen_server_target_qps, experiment_entries):
    """
    This function constructs experiment entries and appends them to the provided list.

    Args:
    scenario: The scenario being tested (e.g. Offline, SingleStream, MultiStream, Server)
    mode_attribs: Attributes of the mode (e.g. AccuracyOnly, PerformanceOnly)
    experiment_tags: Tags associated with the experiment (e.g. loadgen_output, classified_imagenet)
    common_attributes: Attributes common to all experiments (e.g. framework, model_name, sut_name)
    __entry__: An object representing the current experiment
    experiment_list_only: Boolean that determines whether to print or append the experiment
    loadgen_server_target_qps: Target queries per second for the server
    experiment_entries: List of experiment entries to append to

    Returns:
    None
    """
    scenario_attributes = get_scenario_attributes(scenario, mode_attribs, __entry__, loadgen_server_target_qps)
    joined_query = get_experiment_query(experiment_tags, common_attributes, mode_attribs, scenario_attributes)
    
    if experiment_list_only:
        print("Generated query = axs byquery", joined_query)
        print("")
    else:
        experiment_entries.append(__entry__.get_kernel().byquery(joined_query, True))
    return experiment_entries

def generate_experiment_entries(sut_name, sut_system_type, program_name, division, framework, model_name, loadgen_dataset_size, loadgen_buffer_size,  experiment_list_only=False, loadgen_server_target_qps=None, __entry__=None):
    """
    This is the main function that generates experiment entries

    Args:
    sut_name: The name of the system under test (e.g. q2_pro_dc)
    sut_system_type: The type of system under test (e.g. edge, datacenter)
    program_name: The name of the program (e.g. image_classification_onnx_loadgen_py)
    division: The division of the experiment (e.g. closed, open)
    framework: The framework being used (e.g. kilt)
    model_name: The name of the model (e.g. resnet50, retinanet, bert-99, bert-99.9)
    loadgen_dataset_size: The size of the dataset used by the loadgen
    loadgen_buffer_size: The buffer size used by the loadgen
    experiment_list_only: Boolean that determines whether to print or append the experiment
    loadgen_server_target_qps: Target queries per second for the server (only used for Server scenario)
    __entry__: An object representing the current experiment

    Returns:
    A list of experiment entries
    """

    scenarios = get_scenarios(sut_system_type, model_name)
    common_attributes, experiment_tags = get_common_attributes_and_experiment_tags(program_name, framework, model_name, sut_name)
    modes = get_modes(division, model_name)
    experiment_entries = []
    
    for scenario in scenarios:
        for mode_attribs in modes:
            scenario_attributes = get_scenario_attributes(scenario, mode_attribs, __entry__, loadgen_server_target_qps)
            joined_query = get_experiment_query(experiment_tags, common_attributes, mode_attribs, scenario_attributes)

            if experiment_list_only:
                print("Generated query = axs byquery", joined_query)
                print("")
            else:
                experiment_entries.append(__entry__.get_kernel().byquery(joined_query, True))
            # experiment_entries += append_experiment_entries(scenario, mode_attribs, experiment_tags, common_attributes, __entry__, experiment_list_only, loadgen_server_target_qps, experiment_entries)
    
    return experiment_entries


# import os
# import sys
# import shutil
# import json

# def make_local_dir(record_entry, path_list):
#     joined_path = record_entry.get_path(path_list)
#     print(f"Creating directory: {joined_path}", file=sys.stderr)
#     os.makedirs(joined_path, exist_ok=True)
#     return joined_path

# def copy_files(src_file_path, dst_file_path):
#     print(f"Copying: {src_file_path} --> {dst_file_path}", file=sys.stderr)
#     shutil.copy(src_file_path, dst_file_path)

# def write_to_file(path, content, mode='w'):
#     with open(path, mode) as fd:
#         fd.write(content)

# # def store_json(data, path):
# #     with open(path, 'w') as f:
# #         json.dump(data, f, indent=4)

# def run_verify(entry, in_dir, verify_script_path, results_dir, compliance_dir, output_dir):
#     result_verify =  entry.call('get', 'run_verify', {
#             "in_dir": in_dir,
#             "verify_script_path": verify_script_path,
#             "results_dir": results_dir,
#             "compliance_dir": compliance_dir,
#             "output_dir": output_dir
#             } )
#     return result_verify

# def run_truncation_script(entry, log_truncation_script_path, submitted_tree_path, submitter, log_backup_path):
#     result_trucation =  entry.call('get', 'run_truncation_script', {
#             "log_truncation_script_path": log_truncation_script_path,
#             "submitted_tree_path": submitted_tree_path,
#             "submitter": submitter,
#             "log_backup_path": log_backup_path
#             } )
#     return result_trucation

# def process_experiment_entry(experiment_entry, division, submitter, readme_template_path, code_path, model_name_dict, __entry__, __record_entry__, sut_dictionary):
#     experiment_name = experiment_entry['experiment_name']
#     sut_name = experiment_entry['sut_name']

#     model_name = model_name_dict.get(sut_name, sut_name)
#     model_path = os.path.join(code_path, model_name)
    
#     experiment_readme_path = os.path.join(model_path, "README.md")
#     shutil.copyfile(readme_template_path, experiment_readme_path)

#     print(f"  Creating model description: {model_name}  -->  {model_path}", file=sys.stderr)

#     sut_data = {
#         'name': sut_name,
#         'division': division,
#         'submitter': submitter,
#         'status': 'submitted',
#         'details': {
#             'code_path': model_path
#         },
#         'results': {}  # You can populate this with any results data you have
#     }

#     sut_dictionary[sut_name] = sut_data

# def lay_out(experiment_entries, division, submitter, record_entry_name, log_truncation_script_path, submission_checker_path, compliance_path, model_name_dict, experiment_entry=None,  __entry__=None, __record_entry__=None):

#     __record_entry__["tags"] = ["laid_out_submission"]
#     __record_entry__.save(record_entry_name)

#     submitted_tree_path = make_local_dir(__record_entry__, 'submitted_tree')
#     submitter_path = make_local_dir(__record_entry__, ['submitted_tree', division, submitter])
#     code_path = make_local_dir(__record_entry__, ['submitted_tree', division, submitter, 'code'])
#     systems_path = make_local_dir(__record_entry__, ['submitted_tree', division, submitter, 'systems'])

#     sut_dictionary = {}
#     dest_dir = __record_entry__.get_path("")
#     experiment_cmd_list = []
#     readme_template_path = __entry__.get_path("README_template.md")

#     for experiment_entry in experiment_entries:
#         process_experiment_entry(experiment_entry, division, submitter, readme_template_path, code_path, model_name_dict, __entry__, __record_entry__, sut_dictionary)

#     log_backup_path = os.path.join(submitted_tree_path, "accuracy_log.bak")
#     result_truncation = run_truncation_script(__entry__, log_truncation_script_path, submitted_tree_path, submitter, log_backup_path)

#     if result_truncation == 0:
#         shutil.rmtree(log_backup_path, ignore_errors=True)
#     else:
#         return

#     for sut_name in sut_dictionary:
#         sut_data = sut_dictionary[sut_name]
#         sut_data['division'] = division
#         sut_data['submitter'] = submitter
#         sut_path = os.path.join(systems_path, sut_name + '.json')
#         print(f"  Creating SUT description: {sut_name}  -->  {sut_path}", file=sys.stderr)
#         store_json(sut_data, sut_path)

#     return __record_entry__

def lay_out(experiment_entries, division, submitter, record_entry_name, log_truncation_script_path, submission_checker_path, compliance_path, model_name_dict,  __entry__=None, __record_entry__=None):

    def make_local_dir( path_list ):

        joined_path  = __record_entry__.get_path( path_list )
        print(f"Creating directory: {joined_path}", file=sys.stderr)
        try:
            os.makedirs( joined_path )
        except:
            pass
        return joined_path
    __record_entry__["tags"] = ["laid_out_submission"]
    __record_entry__.save( record_entry_name )

    submitted_tree_path = make_local_dir('submitted_tree')

    submitter_path      = make_local_dir( ['submitted_tree', division, submitter ] )
    code_path           = make_local_dir( ['submitted_tree', division, submitter, 'code'] )
    systems_path        = make_local_dir( ['submitted_tree', division, submitter, 'systems'] )

    sut_dictionary      = {}
    dest_dir            = __record_entry__.get_path( "" )
    experiment_cmd_list = []
    readme_template_path = __entry__.get_path("README_template.md")
    print(f"-------------------{len(experiment_entries)}-------------")

    # TO BE REMOVED!
    # experiment_entries = [__entry__.get_kernel().byname("generated_by_retinanet_kilt_loadgen_qaic_on_run_a21d52a914b64df1a2402fffe591fb87")]
    
    for experiment_entry in tqdm(experiment_entries):

        experiment_parameters = []

        src_dir        = experiment_entry.get_path("")
        sut_name       = experiment_entry.get('sut_name')
        sut_data       = experiment_entry.get('sut_data')
        loadgen_mode   = experiment_entry.get('loadgen_mode')
        readme_path    = experiment_entry.get('program_entry').get_path("README.md")
        experiment_cmd = experiment_entry.get('produced_by')
        compliance_test_name       = experiment_entry.get('loadgen_compliance_test')

        experiment_program_name  = experiment_entry.get('program_name')
        benchmark_framework_list = experiment_entry.get('program_name').replace("_loadgen_py", "").split("_")

        framework = benchmark_framework_list[2].upper().replace("RUNTIME","")
        benchmark = benchmark_framework_list[0].title() + " " + benchmark_framework_list[1].title()

        mode = loadgen_mode.replace("Only", "")

        sut_dictionary[sut_name] = sut_data

        print(f"Experiment: {experiment_entry.get_name()} living in {src_dir}", file=sys.stderr)

        model_name  = experiment_entry['model_name']
        mlperf_model_name = get_mlperf_model_name(model_name_dict, model_name)
        if mlperf_model_name is not None:
            display_model_name = mlperf_model_name
        else:
            display_model_name  = model_name

        code_model_program_path        = make_local_dir( [code_path, display_model_name , experiment_program_name ] )
        scenario    = experiment_entry['loadgen_scenario'].lower()

        if os.path.exists(readme_path):
            shutil.copy(readme_path, code_model_program_path)         
        path_readme = os.path.join(code_model_program_path, "README.md")

        # ----------------------------[ measurements ]------------------------------------
        measurement_path = make_local_dir( ['submitted_tree', division, submitter, 'measurements', sut_name, display_model_name, scenario] )
        path_model_readme = os.path.join(measurement_path, "README.md")
        if os.path.exists(readme_template_path):
            with open(readme_template_path, "r") as input_fd:
                template = input_fd.read()
            with open(path_model_readme, "w") as output_fd:
                output_fd.write( template.format(benchmark=benchmark, framework=framework ) )

        with open(path_model_readme, "a") as fd:
            fd.write( "## Benchmarking " + model_name + " model " + "in " + mode + " mode" + "\n" + "```" + "\n" + experiment_cmd + "\n" + "```" + "\n\n")
        print("")
        for src_file_path in ( experiment_entry['loadgen_mlperf_conf_path'], os.path.join(src_dir, 'user.conf') ):
            filename = os.path.basename( src_file_path )
            dst_file_path = os.path.join(measurement_path, filename)
            print(f"    Copying: {src_file_path}  -->  {dst_file_path}", file=sys.stderr)
            shutil.copy( src_file_path, dst_file_path)

        # print("dsfdsf", experiment_entry.get("program_name"))
        # print("experiment_program_name", experiment_program_name)
        # program_name            = experiment_entry.get("program_name", experiment_program_name)
        measurements_meta_path  = os.path.join(measurement_path, f"{sut_name}_{experiment_program_name}_{scenario}.json") 
        print("measurements_meta_path", measurements_meta_path)
        # print("experiment_entry", experiment_entry)
        
        try:
            measurements_meta_data  = {
                "retraining": experiment_entry.get("retraining", ("yes" if experiment_entry.get('retrained', False) else "no")),
                "input_data_types": "int32", #experiment_entry.get("input_data_types"),
                "weight_data_types":  "int8",#experiment_entry.get("weight_data_types"),
                "starting_weights_filename": "https://www.dropbox.com/s/jo92dsoted1ha5q/resnet50_v1.pb", #experiment_entry.get("starting_weights_filename"),
                "weight_transformations": "quantized",#experiment_entry.get("weight_transformations"),
            }
        except KeyError as e:
            raise ValueError(f"\n\n\n ERROR: Key {e} is missing from experiment_entry \n\n")

        store_json(measurements_meta_data, measurements_meta_path)

        experiment_entry.parent_objects = None

        # --------------------------------[ results ]--------------------------------------
        mode        = {
            'AccuracyOnly': 'accuracy',
            'PerformanceOnly': 'performance',
        }[ experiment_entry['loadgen_mode'] ]

        if  ( mode== 'accuracy') or ( mode == 'performance' and compliance_test_name is False ):
            results_path_syll   = ['submitted_tree', division, submitter, 'results', sut_name, display_model_name, scenario, mode]
        elif compliance_test_name  in [ "TEST01", "TEST04", "TEST05" ]:
            results_path_syll = ['submitted_tree', division, submitter, 'compliance', sut_name , display_model_name, scenario , compliance_test_name ]
            if compliance_test_name == "TEST01":
                results_path_syll_TEST01_acc = ['submitted_tree', division, submitter, 'compliance', sut_name , display_model_name, scenario , compliance_test_name, 'accuracy' ]
                results_path_TEST01_acc = make_local_dir(results_path_syll_TEST01_acc)
            else:
                results_path = make_local_dir(results_path_syll)

        files_to_copy       = [ 'mlperf_log_summary.txt', 'mlperf_log_detail.txt' ]

        if mode=='accuracy' or compliance_test_name == "TEST01":
            files_to_copy.append( 'mlperf_log_accuracy.json' )
        if mode=='performance' and compliance_test_name is False:
            results_path_syll.append( 'run_1' )
        if mode=='performance' and compliance_test_name in [ "TEST01", "TEST04", "TEST05" ]:
            results_path_syll.extend(( mode, 'run_1' ))

        results_path        = make_local_dir( results_path_syll )
        
        for filename in files_to_copy:
            src_file_path = os.path.join(src_dir, filename)
            print(f"-------------mode:{mode}-------compliance_test_name{compliance_test_name}-----------------")
            print("source file path", src_file_path, "destination file path", dst_file_path)

            if (compliance_test_name == "TEST01" and filename == 'mlperf_log_accuracy.json'):
                dst_file_path = os.path.join(results_path_TEST01_acc, filename)
            else:
                dst_file_path = os.path.join(results_path, filename)

            print(f"    Copying: {src_file_path}  -->  {dst_file_path}", file=sys.stderr)
            shutil.copy( src_file_path, dst_file_path)
        
        if mode=='accuracy' or compliance_test_name == "TEST01":
            if experiment_program_name in ["object_detection_onnx_loadgen_py", "retinanet_kilt_loadgen_qaic"]:
                accuracy_content    = str(experiment_entry["accuracy_report"])
            elif experiment_program_name in [ "bert_squad_onnxruntime_loadgen_py", "bert_squad_kilt_loadgen_c",  "bert_squad_kilt_loadgen_qaic"]:
                accuracy_content    = str(experiment_entry["accuracy_report"])
            elif experiment_program_name in ["image_classification_onnx_loadgen_py", "image_classification_torch_loadgen_py","resnet50_kilt_loadgen_qaic"]:

                accuracy_content    = str(experiment_entry["accuracy_report"])
            if mode == 'accuracy':
                dst_file_path       = os.path.join(results_path, "accuracy.txt")
            elif compliance_test_name == "TEST01":
                dst_file_path       = os.path.join(results_path_TEST01_acc, "accuracy.txt")
            print(f"    Storing accuracy -->  {dst_file_path}", file=sys.stderr)
            with open(dst_file_path, "w") as fd:
                if mode=='accuracy':
                    fd.write(accuracy_content + "\n")

        # -------------------------------[ compliance , verification ]--------------------------------------
        if compliance_test_name in [ "TEST01", "TEST04", "TEST05" ]:
            compliance_path_test = make_local_dir( ['submitted_tree', division, submitter, 'compliance', sut_name , display_model_name, scenario, compliance_test_name ] )

            print("Verification for ", compliance_test_name)

            tmp_dir = make_local_dir( ['submitted_tree', division, submitter, 'compliance', sut_name , display_model_name, scenario, 'tmp' ] )
            results_dir = os.path.join(submitter_path , 'results', sut_name, display_model_name, scenario)
            compliance_dir = src_dir
            output_dir = os.path.join(submitter_path ,'compliance', sut_name , display_model_name, scenario)
            verify_script_path =  os.path.join(compliance_path,compliance_test_name, "run_verification.py")
            result_verify =  __entry__.call('get', 'run_verify', {
                    "in_dir": tmp_dir,
                    "verify_script_path": verify_script_path,
                    "results_dir": results_dir,
                    "compliance_dir": compliance_dir,
                    "output_dir": output_dir
                        } )
            print("###############")
            print("result_verify", result_verify)
            print("result_verify", type(result_verify))
            print("-----------------------")
            # shutil.rmtree(tmp_dir, ignore_errors=True)
            if result_verify == "":
                shutil.rmtree(tmp_dir, ignore_errors=True)
            # else:
            #     return

    print(f"Truncating logs in:  {src_dir}", file=sys.stderr)
    log_backup_path     = os.path.join(submitted_tree_path, "accuracy_log.bak")

    result_trucation =  __entry__.call( 'get', 'run_truncation_script', {
            "log_truncation_script_path": log_truncation_script_path,
            "submitted_tree_path": submitted_tree_path,
            "submitter": submitter,
            "log_backup_path": log_backup_path
            } )

    if  result_trucation == 0:
        shutil.rmtree(log_backup_path, ignore_errors=True)
    else:
        return
    # -------------------------------[ systems ]--------------------------------------
    for sut_name in sut_dictionary:
        sut_data = sut_dictionary[sut_name]

        sut_data['division']    = division
        sut_data['submitter']   = submitter

        sut_path = os.path.join( systems_path, sut_name+'.json' )

        print(f"  Creating SUT description: {sut_name}  -->  {sut_path}", file=sys.stderr)
        store_json(sut_data, sut_path)

    return __record_entry__

def run_checker(submission_checker_path, submitted_tree_path, submitter, division, __entry__):

    checker_log_path  = os.path.join(submitted_tree_path, division, submitter )
    result_checker =  __entry__.call( 'get', 'run_checker_script', {
            "submission_checker_path": submission_checker_path,
            "submitted_tree_path": submitted_tree_path
             } )

    print(result_checker)
    logfile = open(os.path.join(checker_log_path,"submission-checker.log"),"w")
    logfile.write(result_checker)


def full_run(experiment_entries, division, submitter, record_entry_name, log_truncation_script_path, submission_checker_path, compliance_path, model_name_dict, __entry__=None, __record_entry__=None):

    __record_entry__["tags"] = ["laid_out_submission"]
    __record_entry__.save( record_entry_name )
    submitted_tree_path  = __record_entry__.get_path( ['submitted_tree'] )

    if os.path.exists(submitted_tree_path):
        print("Run checker...")
        run_checker(submission_checker_path, submitted_tree_path,  submitter, division, __entry__)
    else:
        print("Run lay_out...")
        lay_out(experiment_entries, division, submitter, record_entry_name, log_truncation_script_path, submission_checker_path, compliance_path, model_name_dict,  __entry__, __record_entry__)
        print("Run checker...")
        run_checker(submission_checker_path, submitted_tree_path, submitter, division, __entry__)

