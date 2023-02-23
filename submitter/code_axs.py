#!/usr/bin/env python3


import json
import os
import shutil
import subprocess
import sys
from typing import ( Dict, List, Tuple )
from pathlib import Path
compliance_test_list = []

def store_json(data_structure, json_file_path):
    json_data   = json.dumps( data_structure , indent=4)

    with open(json_file_path, "w") as json_fd:
        json_fd.write( json_data+"\n" )

def generate_experiment_entries(sut_system_type, program_name, division, framework, model_name, loadgen_dataset_size, loadgen_buffer_size, __entry__=None):

    if sut_system_type == "edge":
        if model_name in ("resnet50", "retinanet_openimages"):
            scenarios = ["Offline", "SingleStream", "MultiStream" ]
        else:
            scenarios = ["Offline", "SingleStream" ]
    elif sut_system_type == "datacentre":
        scenarios = ["Offline", "Server" ]
    common_attributes = {}
    if program_name in ("image_classification_onnx_loadgen_py", "image_classification_torch_loadgen_py"):
        experiment_tags = [ "loadgen_output", "classified_imagenet" ]
        common_attributes["loadgen_dataset_size"] = 50000
        common_attributes["loadgen_buffer_size"]  = 1024
    elif program_name == "object_detection_onnx_loadgen_py":
        experiment_tags = [ "loadgen_output", "detected_coco" ]
        if model_name == "retinanet_openimages":
            common_attributes["loadgen_dataset_size"] = 24781
            common_attributes["loadgen_buffer_size"]  = 64
        elif model_name == "retinanet_coco":
            common_attributes["loadgen_dataset_size"] = 5000
            common_attributes["loadgen_buffer_size"]  = 64

    elif  program_name == "bert_squad_onnxruntime_loadgen_py":
        experiment_tags = [ "loadgen_output", "bert_squad" ]
        common_attributes["loadgen_dataset_size"] = 10833
        common_attributes["loadgen_buffer_size"]  = 10833

    common_attributes["framework"] = framework
    common_attributes["model_name"] = model_name

    modes = [
        [ "loadgen_mode=AccuracyOnly" ],
        [ "loadgen_mode=PerformanceOnly", "loadgen_compliance_test-" ],
    ]
    if division == "closed":
        if model_name == "resnet50":
            compliance_test_list = [ 'TEST01', 'TEST04', 'TEST05' ]
        elif program_name in ( "bert_squad_onnxruntime_loadgen_py", "bert_squad_qaic_loadgen_kilt" ):
            compliance_test_list = [ 'TEST01', 'TEST05' ]
        elif program_name == "object_detection_onnx_loadgen_py" and model_name == "retinanet_openimages":
            compliance_test_list = [ 'TEST01', 'TEST05' ]
        else:
            compliance_test_list = []

        for compliance_test_name in compliance_test_list:
            modes.append( [ "loadgen_mode=PerformanceOnly", "loadgen_compliance_test="+compliance_test_name ] )

    experiment_entries = []
    for sc in scenarios:
        scenario_attributes = { "loadgen_scenario": sc }
        for mode_attribs in modes:
            list_output = []
            if "loadgen_mode=AccuracyOnly" in mode_attribs:
                if sc == "Server":
                     scenario_attributes["loadgen_target_qps"] = __entry__["loadgen_target_qps"]
            elif "loadgen_mode=PerformanceOnly" in mode_attribs:
                if sc in ("Offline", "Server"):
                    scenario_attributes["loadgen_target_qps"] = __entry__["loadgen_target_qps"]
                elif sc in ("SingleStream", "MultiStream"):
                    scenario_attributes[ "loadgen_target_latency" ] = __entry__["loadgen_target_latency"]
                elif  sc == "MultiStream":
                    scenario_attributes["loadgen_multistreamness"] = __entry__["loadgen_multistreamness"]

            list_query = ( experiment_tags +
                [ f"{k}={common_attributes[k]}" for k in common_attributes ] +
                mode_attribs +
                [ f"{k}={scenario_attributes[k]}" for k in scenario_attributes ]   )

            joined_query = ','.join( list_query )

            #print("Generated query = ", joined_query )
            #print("")
            experiment_entries.append(__entry__.get_kernel().byquery(joined_query, True))

    return experiment_entries


def lay_out(experiment_entries, division, submitter, record_entry_name, log_truncation_script_path, submission_checker_path, compliance_path, __entry__=None, __record_entry__=None):

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
    for experiment_entry in experiment_entries:
        experiment_parameters = []

        model_entry      = experiment_entry["model_entry"]
        model_entry_path = model_entry.get_path("")
        with  open(os.path.join(model_entry_path, "data_axs.json")) as file_json:
            model_dict = json.load(file_json)
        keys_list = ["input_data_types", "weight_data_types", "url", "weight_transformations"]
        for key in keys_list:
            if key not in model_dict:
                print("Error: Some of the following parameters (input_data_types, weight_data_types, url, weight_transformations, retrained) are not present in model file.")
                return

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
        if experiment_program_name == "object_detection_onnx_loadgen_py":
            display_model_name  = model_name.replace('_', '-')      # replaces ssd_resnet34 with ssd-resnet34
        elif model_name == "bert_large":
            display_model_name  = "bert-99"
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

        for src_file_path in ( experiment_entry['loadgen_mlperf_conf_path'], os.path.join(src_dir, 'user.conf') ):
            filename = os.path.basename( src_file_path )
            dst_file_path = os.path.join(measurement_path, filename)
            print(f"    Copying: {src_file_path}  -->  {dst_file_path}", file=sys.stderr)
            shutil.copy( src_file_path, dst_file_path)

        program_name            = experiment_entry.get("program_name", experiment_program_name)
        measurements_meta_path  = os.path.join(measurement_path, f"{sut_name}_{program_name}_{scenario}.json") 
        measurements_meta_data  = {
            "retraining": ("yes" if model_entry.get('retrained', False) else "no"),
            "input_data_types": model_entry["input_data_types"],
            "weight_data_types": model_entry["weight_data_types"],
            "starting_weights_filename": model_entry["url"],
            "weight_transformations": model_entry["weight_transformations"],
        }
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
            results_path_syll = ['submitted_tree', division, submitter, 'compliance', sut_name , model_name, scenario , compliance_test_name ]
            if compliance_test_name == "TEST01":
                results_path_syll_TEST01_acc = ['submitted_tree', division, submitter, 'compliance', sut_name , model_name, scenario , compliance_test_name, 'accuracy' ]
                results_path_TEST01_acc = make_local_dir(results_path_syll_TEST01_acc)

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

            if (compliance_test_name == "TEST01" and filename == 'mlperf_log_accuracy.json'):
                dst_file_path = os.path.join(results_path_TEST01_acc, filename)
            else:
                dst_file_path = os.path.join(results_path, filename)

            print(f"    Copying: {src_file_path}  -->  {dst_file_path}", file=sys.stderr)
            shutil.copy( src_file_path, dst_file_path)

        if mode=='accuracy' or compliance_test_name == "TEST01":
            if experiment_program_name == "object_detection_onnx_loadgen_py":
                accuracy_content    = str(experiment_entry["mAP"])
            elif experiment_program_name == "bert_squad_onnxruntime_loadgen_py":
                accuracy_content    = str(experiment_entry["f1"])
            elif experiment_program_name == "image_classification_onnx_loadgen_py" or experiment_program_name == "image_classification_torch_loadgen_py":

                accuracy_content    = str(experiment_entry["accuracy_report"])
            if mode == 'accuracy':
                dst_file_path       = os.path.join(results_path, "accuracy.txt")
            elif compliance_test_name == "TEST01":
                dst_file_path       = os.path.join(results_path_TEST01_acc, "accuracy.txt")
            print(f"    Storing accuracy -->  {dst_file_path}", file=sys.stderr)
            with open(dst_file_path, "w") as fd:
                fd.write(accuracy_content + "\n")

        # -------------------------------[ compliance , verification ]--------------------------------------
        if compliance_test_name in [ "TEST01", "TEST04", "TEST05" ]:
            compliance_path_test = make_local_dir( ['submitted_tree', division, submitter, 'compliance', sut_name , model_name, scenario, compliance_test_name ] )

            ("Verification for ", compliance_test_name)

            tmp_dir = make_local_dir( ['submitted_tree', division, submitter, 'compliance', sut_name , model_name, scenario] )
            results_dir = os.path.join(submitter_path , 'results', sut_name, display_model_name, scenario)
            compliance_dir = src_dir
            output_dir = os.path.join(submitter_path ,'compliance', sut_name , model_name, scenario)
            result = subprocess.run(
            [
                sys.executable,
                os.path.join(compliance_path,compliance_test_name, "run_verification.py"),
                "--results_dir",
                results_dir,
                "--compliance_dir",
                compliance_dir,
                "--output_dir",
                output_dir,
            ],
                cwd=tmp_dir,
            )

    print(f"Truncating logs in:  {src_dir}", file=sys.stderr)
    log_backup_path     = os.path.join(submitted_tree_path, "accuracy_log.bak")
    truncation_cmd = [
        sys.executable,
        log_truncation_script_path,
        '--input',
        submitted_tree_path,
        '--submitter',
        submitter,
        '--backup',
        log_backup_path
    ]
    print('Truncation cmd:\n\t' + ' '.join(truncation_cmd))
    subprocess.run( truncation_cmd )
    shutil.rmtree(log_backup_path, ignore_errors=True)

    # -------------------------------[ systems ]--------------------------------------
    for sut_name in sut_dictionary:
        sut_data = sut_dictionary[sut_name]

        sut_data['division']    = division
        sut_data['submitter']   = submitter

        sut_path = os.path.join( systems_path, sut_name+'.json' )

        print(f"  Creating SUT description: {sut_name}  -->  {sut_path}", file=sys.stderr)
        store_json(sut_data, sut_path)

    args = [sys.executable, submission_checker_path, "--input", submitted_tree_path, "--csv", "/dev/null"]

    res = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = res.communicate()
    print(stderr.decode())

    checker_log_path = make_local_dir( ['submitted_tree', division, submitter ] )
    logfile = open(os.path.join(checker_log_path,"submission-checker.log"),"w")
    logfile.write(stderr.decode())

    return __record_entry__
