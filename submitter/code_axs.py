#!/usr/bin/env python3

import os
import shutil
import sys
from ufun import save_json

def get_mlperf_model_name(model_name_dict, model_name):
    if model_name in model_name_dict.keys():
        return model_name_dict[model_name]
    else:
        return None

def task_from_program_name(program_name):

    if program_name.startswith("image_classification_") or program_name.startswith("resnet50_"):
        return "image_classification"
    elif program_name.startswith("object_detection_") or program_name.startswith("retinanet_"):
        return "object_detection"
    elif program_name.startswith("bert_squad_"):
        return "bert_squad"
    elif program_name.startswith("gptj_cnndm_"):
        return "gptj_cnndm"
    else:
        return "UNKNOWN"


def generate_experiment_entries( power, sut_name, sut_system_type, program_name, division, model_name, experiment_tags, framework, device, loadgen_dataset_size, loadgen_buffer_size, experiment_list_only=False, scenarios=None, extra_common_attributes=None, per_scenario_attributes=None, __entry__=None):

    task = task_from_program_name(program_name)

    if not scenarios:
        if sut_system_type == "edge":
            if model_name in ("resnet50", "retinanet_openimages"):
                scenarios = ["Offline", "SingleStream", "MultiStream" ]
            else:
                scenarios = ["Offline", "SingleStream" ]
        elif sut_system_type in ("dc", "datacenter"):
            scenarios = ["Offline", "Server" ]

    common_attributes = {
        "sut_name":             sut_name,
        "model_name":           model_name,
        "framework":            framework,
    }
    if framework=="kilt":
        common_attributes["device"]     = device

    if framework=="kilt" and task in ("image_classification", "object_detection"):
        common_attributes["first_n"]    = loadgen_dataset_size
    else:
        common_attributes["loadgen_dataset_size"]   = loadgen_dataset_size
        common_attributes["loadgen_buffer_size"]    = loadgen_buffer_size

    extra_common_attributes = extra_common_attributes or {}
    per_scenario_attributes = per_scenario_attributes or {}

    modes = [
        [ "loadgen_mode=AccuracyOnly" ],
        [ "loadgen_mode=PerformanceOnly", "loadgen_compliance_test-" ],
    ]

    if division == "closed":
        compliance_test_list = {
            "image_classification": [ 'TEST01', 'TEST04', 'TEST05' ],
            "object_detection":     [ 'TEST01', 'TEST05' ],
            "bert_squad":           [ 'TEST01', 'TEST05' ],
            "gptj_cnndm":           [ ],
        }[task]

        for compliance_test_name in compliance_test_list:
            modes.append( [ "loadgen_mode=PerformanceOnly", "loadgen_compliance_test="+compliance_test_name ] )

    experiment_entries = []
    for sc in scenarios:
        scenario_specific_attributes = per_scenario_attributes.get(sc, {})

        scenario_attributes = { "loadgen_scenario": sc }
        for mode_attribs in modes:
            list_output = []

            if power:
                if ("loadgen_mode=PerformanceOnly" in mode_attribs) and ("loadgen_compliance_test-") in mode_attribs:
                    experiment_tags[0]="power_loadgen_output"
                else:
                    experiment_tags[0]="loadgen_output"

            list_query = ( experiment_tags +
                [ f"{k}={common_attributes[k]}" for k in common_attributes ] +
                [ f"{k}={extra_common_attributes[k]}" for k in extra_common_attributes ] +
                [ f"{k}={scenario_specific_attributes[k]}" for k in scenario_specific_attributes ] +
                mode_attribs +
                [ f"{k}={scenario_attributes[k]}" for k in scenario_attributes ]   )

            joined_query = ','.join( list_query )
            if experiment_list_only:
                print("Generated query = ", joined_query )
                print("")
            else:
                experiment_entries.append(__entry__.get_kernel().byquery(joined_query, True))

    return experiment_entries

def lay_out(experiment_entries, division, submitter, sut_path, record_entry_name, log_truncation_script_path, submission_checker_path, compliance_path, model_name_dict, model_meta_data=None,  __entry__=None, __record_entry__=None):

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

    sut_descriptions_dictionary      = {}
    dest_dir            = __record_entry__.get_path( "" )
    experiment_cmd_list = []
    readme_template_path = __entry__.get_path("README_template.md")

    for experiment_entry in experiment_entries:
        experiment_parameters = []
        if "power_loadgen_output" in experiment_entry["tags"]:
            power_experiment_entry = experiment_entry
            last_mlperf_logs_path = power_experiment_entry.get_path("last_mlperf_logs")
            origin_experiment_path = os.readlink(last_mlperf_logs_path)
            origin_experiment_name = origin_experiment_path.split("/")[-1]
            experiment_entry = __entry__.get_kernel().byname(origin_experiment_name)

        src_dir         = experiment_entry.get_path("")
        sut_name        = experiment_entry.get('sut_name')
        sut_description = experiment_entry.get('sut_description')
        loadgen_mode    = experiment_entry.get('loadgen_mode')

        with_power      = experiment_entry.get("with_power")

        experiment_program_name  = experiment_entry.get('program_name')
        program_entry = __entry__.get_kernel().byname(experiment_program_name)
        readme_path    = program_entry.get_path("README.md")

        experiment_cmd = experiment_entry.get('produced_by')
        compliance_test_name       = experiment_entry.get('loadgen_compliance_test')

        framework       = experiment_entry.get('framework').upper()

        experiment_task     = task_from_program_name(experiment_program_name)
        display_benchmark   = experiment_task.replace("_", " ").title()

        mode = loadgen_mode.replace("Only", "")

        sut_descriptions_dictionary[sut_name] = sut_description

        print(f"Experiment: {experiment_entry.get_name()} living in {src_dir}", file=sys.stderr)

        model_name  = experiment_entry['model_name']
        mlperf_model_name = get_mlperf_model_name(model_name_dict, model_name)
        if mlperf_model_name:
            display_model_name = mlperf_model_name
        else:
            display_model_name  = model_name

        modified_program_name   = experiment_program_name.replace("resnet50", "image_classification")
        code_model_program_path = make_local_dir( [code_path, display_model_name , modified_program_name ] )
        scenario    = experiment_entry['loadgen_scenario'].lower()

        if os.path.exists(readme_path):
            print(f"    Copying: {readme_path}  -->  {code_model_program_path}", file=sys.stderr)
            shutil.copy(readme_path, code_model_program_path)         
        else:
            print(f"    NOT Copying: {readme_path}  -->  {code_model_program_path}", file=sys.stderr)
        path_readme = os.path.join(code_model_program_path, "README.md")

        # ----------------------------[ measurements ]------------------------------------
        measurement_general_path = make_local_dir ( ['submitted_tree', division, submitter, 'measurements', sut_name ] )
        measurement_path = make_local_dir( ['submitted_tree', division, submitter, 'measurements', sut_name, display_model_name, scenario] )

        path_model_readme = os.path.join(measurement_path, "README.md")
        if os.path.exists(readme_template_path):
            with open(readme_template_path, "r") as input_fd:
                template = input_fd.read()
            with open(path_model_readme, "w") as output_fd:
                output_fd.write( template.format(benchmark=display_benchmark, framework=framework ) )

        with open(path_model_readme, "a") as fd:
            fd.write( "## Benchmarking " + model_name + " model " + "in " + mode + " mode" + "\n" + "```" + "\n" + experiment_cmd + "\n" + "```" + "\n\n")
        print("")
        for src_file_path in ( experiment_entry['loadgen_mlperf_conf_path'], os.path.join(src_dir, 'user.conf') ):

            filename = os.path.basename( src_file_path )
            dst_file_path = os.path.join(measurement_path, filename)
            print(f"    Copying: {src_file_path}  -->  {dst_file_path}", file=sys.stderr)
            shutil.copy( src_file_path, dst_file_path)

        measurements_meta_path  = os.path.join(measurement_path, f"{sut_name}_{modified_program_name}_{scenario}.json")

        # model_meta_data has become a generic source of measurements_meta_data (can be overridden, can come from the model, or be spread through the experiment entry)
        model_meta_data = model_meta_data or experiment_entry.get("compiled_model_source_entry", experiment_entry)

        try:
            measurements_meta_data  = {
                "retraining": model_meta_data.get("retraining", ("yes" if model_meta_data.get('retrained', False) else "no")),
                "input_data_types": model_meta_data["input_data_types"],
                "weight_data_types": model_meta_data["weight_data_types"],
                "starting_weights_filename": model_meta_data["url"],
                "weight_transformations": model_meta_data["weight_transformations"],
            }
        except KeyError as e:
            print(f"Key {e} is missing from model_meta_data or the model")
            return

        save_json(measurements_meta_data, measurements_meta_path, indent=4)

        experiment_entry.parent_objects = None

        if with_power:
            analyzer_table_file = "analyzer_table.md"
            power_settings_file = "power_settings.md"

            sut_analyzer_table_path = os.path.join(sut_path, analyzer_table_file)
            sut_power_settings_path = os.path.join(sut_path, power_settings_file)

            analyzer_table_file_path = os.path.join(measurement_general_path, analyzer_table_file)
            power_settings_file_path = os.path.join(measurement_general_path, power_settings_file)

            if os.path.isfile(sut_analyzer_table_path ) and os.path.isfile(sut_power_settings_path):
                shutil.copy2(sut_analyzer_table_path, analyzer_table_file_path)
                shutil.copy2(sut_power_settings_path, power_settings_file_path)
        # --------------------------------[ results ]--------------------------------------
        mode        = {
            'AccuracyOnly': 'accuracy',
            'PerformanceOnly': 'performance',
        }[ experiment_entry['loadgen_mode'] ]

        if  ( mode== 'accuracy') or ( mode == 'performance' and not compliance_test_name):
            results_path_syll   = ['submitted_tree', division, submitter, 'results', sut_name, display_model_name, scenario, mode]
        elif compliance_test_name  in [ "TEST01", "TEST04", "TEST05" ]:
            results_path_syll = ['submitted_tree', division, submitter, 'compliance', sut_name , display_model_name, scenario , compliance_test_name ]
            if compliance_test_name == "TEST01":
                results_path_syll_TEST01_acc = ['submitted_tree', division, submitter, 'compliance', sut_name , display_model_name, scenario , compliance_test_name, 'accuracy' ]
                results_path_TEST01_acc = make_local_dir(results_path_syll_TEST01_acc)

        files_to_copy       = [ 'mlperf_log_summary.txt', 'mlperf_log_detail.txt' ]

        if mode=='accuracy' or compliance_test_name == "TEST01":
            files_to_copy.append( 'mlperf_log_accuracy.json' )
        if mode=='performance' and not compliance_test_name:
            if not with_power:
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

            if not with_power:
                print(f"    Copying: {src_file_path}  -->  {dst_file_path}", file=sys.stderr)
                shutil.copy( src_file_path, dst_file_path)

        if with_power and mode=='performance' and not compliance_test_name:
             power_src_dir = power_experiment_entry.get_path("power_logs")
             dir_list = ['power', 'ranging', 'run_1']

             for elem in dir_list:
                results_path_syll.append( elem )
                src_file_path = power_src_dir + "/" + elem + "/"

                results_path        = make_local_dir( results_path_syll )

                for file_name in os.listdir(src_file_path):
                    if file_name != "ptd_out.txt":
                        src_file_path_file = src_file_path + file_name
                        results_path_file = results_path + "/" + file_name
                        shutil.copy(src_file_path_file, results_path_file)
                results_path_syll.remove(elem)

        if mode=='accuracy' or compliance_test_name == "TEST01":
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

            ("Verification for ", compliance_test_name)

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
            if result_verify == "":
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                return
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
    for sut_name in sut_descriptions_dictionary:
        sut_description = sut_descriptions_dictionary[sut_name]

        sut_description['division']    = division
        sut_description['submitter']   = submitter
        if sut_description['system_type'] == 'dc':
            sut_description['system_type'] = 'datacenter'

        sut_path = os.path.join( systems_path, sut_name+'.json' )

        print(f"  Creating SUT description: {sut_name}  -->  {sut_path}", file=sys.stderr)
        save_json(sut_description, sut_path, indent=4)

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


def full_run(experiment_entries, division, submitter, record_entry_name, log_truncation_script_path, submission_checker_path, sut_path, compliance_path, model_name_dict, model_meta_data=None, __entry__=None, __record_entry__=None):

    __record_entry__["tags"] = ["laid_out_submission"]
    __record_entry__.save( record_entry_name )
    submitted_tree_path  = __record_entry__.get_path( ['submitted_tree'] )

    if os.path.exists(submitted_tree_path):
        print("Run checker...")
        run_checker(submission_checker_path, submitted_tree_path,  submitter, division, __entry__)
    else:
        print("Run lay_out...")
        lay_out(experiment_entries, division, submitter, sut_path, record_entry_name, log_truncation_script_path, submission_checker_path, compliance_path, model_name_dict, model_meta_data,  __entry__, __record_entry__)
        print("Run checker...")
        run_checker(submission_checker_path, submitted_tree_path, submitter, division, __entry__)

