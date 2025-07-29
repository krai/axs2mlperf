#!/usr/bin/env python3

import os
import shutil
import sys
import json
from ufun import save_json
from tabulate import tabulate

def create_run_verification_input_dict(task, tmp_dir, verify_script_path, results_dir, compliance_dir, output_dir, scenario, dtype=None):
    result_dict = {
        "in_dir": tmp_dir,
        "verify_script_path": verify_script_path,
        "compliance_dir": compliance_dir,
        "output_dir": output_dir
    }
    if task in ["llm", "llama2", "llama3_1", "moe"]:
        result_dict [ "scenario" ] = scenario
        result_dict [ "dtype" ] = dtype
    else:
        result_dict [ "results_dir" ] = results_dir
    print("result_dict = ", result_dict)
    return result_dict


def scenarios_from_sut_type_and_task(sut_system_type, task):

    if sut_system_type == "edge":
        if task in ("image_classification", "object_detection"):
            scenarios = ["Offline", "SingleStream", "MultiStream" ]
        else:
            scenarios = ["Offline", "SingleStream" ]
    elif sut_system_type in ("dc", "datacenter"):
        scenarios = ["Offline", "Server" ]

    return scenarios


def list_experiment_entries( power, sut_name, sut_system_type, task, division, experiment_tags, framework, device, loadgen_dataset_size, loadgen_buffer_size, scenarios, model_name=None, mlperf_model_name=None, generate=False, infer_from_ss=False, extra_common_attributes=None, per_scenario_attributes=None, require_compliance=None, substitution_map=None, __entry__=None):
    """Generate a list of entries that are expected to be used for a particular submission.
        --generate+ enforces immediate creation of those entries (off by default)

        --division=close among other things requires compliance tests to be run ; --require_compliance- overrides this.
        --division=open among other things does not require compliance tests ; --require_compliance+ overrides this.

Usage examples:

    # This submission is supposed to be complete on its own: {[nonInteractive_]Server,Offline} x {Accuracy,Performance,TEST06}
    axs byname submitter , list_experiment_entries  --framework=openai --task=llama2  --division=open --require_compliance+ --sut_name=xd670_h200_x8_sglang --program_name=llama2_using_openai_loadgen --sut_system_type=datacenter --submitter=Krai  --submission_entry_name=laid_out_sglang --extra_common_attributes,::=mlperf_model_name:llama3_1-70b-fp8_pre

    # While this one has 3 explicit experiments: {[Interactive_]Server} x {Accuracy,Performance,TEST06}, and we want to Infer (import with substitution) Offline from the previous group.
    axs byname submitter , list_experiment_entries  --framework=openai --task=llama2  --division=open --require_compliance+ --sut_name=xd670_h200_x8_sglang --program_name=llama2_using_openai_loadgen --sut_system_type=datacenter --submitter=Krai  --submission_entry_name=laid_out_sglang --extra_common_attributes,::=mlperf_model_name:llama3_1-70b-interactive-fp8_pre

    # We use --substitution_map to infer Offline experiments from another mlperf_model_name while using "own" Server experiments:
    axs byname submitter , list_experiment_entries  --framework=openai --task=llama2  --division=open --require_compliance+ --sut_name=xd670_h200_x8_sglang --program_name=llama2_using_openai_loadgen --sut_system_type=datacenter --submitter=Krai  --submission_entry_name=laid_out_sglang --extra_common_attributes,::=mlperf_model_name:llama3_1-70b-interactive-fp8_pre ---substitution_map='{"mlperf_model_name":{"llama3_1-70b-interactive-fp8_pre":"llama3_1-70b-fp8_pre"}}'
    """

    if infer_from_ss:
        substitution_map = {
            "loadgen_scenario": {
                "Offline":      "SingleStream",
                "MultiStream":  "SingleStream"
            }
        }
    elif substitution_map is None:
        substitution_map = {}

    common_attributes = {
        "framework":            framework,
        "task":                 task,
        "sut_name":             sut_name,
#        "loadgen_dataset_size": loadgen_dataset_size,
#        "loadgen_buffer_size":  loadgen_buffer_size,
    }
    if model_name:
        common_attributes["model_name"] = model_name
    if mlperf_model_name:
        common_attributes["mlperf_model_name"] = mlperf_model_name
    if framework=="kilt":
        common_attributes["device"]     = device

    extra_common_attributes = extra_common_attributes or {}
    per_scenario_attributes = per_scenario_attributes or {}

    modes = [
        [ "loadgen_mode=AccuracyOnly" ],
        [ "loadgen_mode=PerformanceOnly", "loadgen_compliance_test-" ],
    ]

    if require_compliance is None:  # but it can be forced as True or False via --require_compliance+ or --require_compliance-
        require_compliance = division == "closed"

    if require_compliance:
        compliance_test_list = {
            "image_classification": [ 'TEST01', 'TEST04' ],
            "object_detection":     [ 'TEST01' ],
            "bert":                 [ 'TEST01' ],
            "gptj":                 [ ],
            "text_to_image":        [ 'TEST01', 'TEST04' ],
            "llm":                  [ 'TEST06' ],
            "llama2":               [ 'TEST06' ],
            "llama3_1":               [ 'TEST06' ],
            "moe":                  [ 'TEST06' ]
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

            candidate_entry = __entry__.get_kernel().byquery(joined_query, False)

            inferrable_case, inferred_entry = False, False
            for substituted_param in substitution_map:
                for target_value in substitution_map[substituted_param]:
                    if f"{substituted_param}={target_value}" in joined_query:
                        source_value    = substitution_map[substituted_param][target_value]
                        inferred_query  = joined_query.replace(f"{substituted_param}={target_value}",f"{substituted_param}={source_value}")
                        inferrable_case = [ substituted_param, target_value, source_value, inferred_query ]    # 4-tuple

            if inferrable_case:
                [ substituted_param, target_value, source_value, inferred_query ] = inferrable_case
                inferred_entry  = __entry__.get_kernel().byquery(inferred_query, False)
                inferrable_case.append( inferred_entry )    # extended to 5-tuple ( still adding None if not found )

            if generate:
                if inferrable_case and not candidate_entry:
                    print(f"Entry {joined_query} is missing, but INFERRABLE, adding it as a mapping\n")
                    [ substituted_param, target_value, source_value, inferred_query, inferred_entry ] = inferrable_case
                    candidate_entry = inferred_entry or __entry__.get_kernel().byquery(inferred_query, True)
                    candidate_entry[substituted_param] = target_value
                else:
                    if candidate_entry:
                        print(f"Entry {joined_query} was already PRESENT, adding it to the list\n")
                    else:
                        print(f"Entry {joined_query} was MISSING and not inferrable, generating it now\n")
                        candidate_entry = candidate_entry or __entry__.get_kernel().byquery(joined_query, True)    # now generating for real

                experiment_entries.append( candidate_entry )

            else:
                if candidate_entry:
                    presence_msg = "Present"
                elif inferrable_case:
                    presence_msg = "Inferred"
                else:
                    presence_msg = "Missing"

                print(f"[{presence_msg}]\t\taxs byquery {joined_query}")
                if candidate_entry:
                    print(f"Present Location:\t\t{candidate_entry.get_path()}")
                elif inferred_entry:
                    print(f"Inferred Location:\t\t{inferred_entry.get_path()}")
                print("")

    return experiment_entries

def make_local_dir( path_list, submitted_tree_path ):

        joined_path  = os.path.join( submitted_tree_path, *path_list )
        print(f"Creating directory: {joined_path}", file=sys.stderr)
        try:
            os.makedirs( joined_path )
        except:
            pass
        return joined_path


def get_testing_entry(experiment_entry):

    entry_path = experiment_entry.get_path("")
    path_to_program_output = os.path.join(entry_path, 'program_output.json')

    with open(path_to_program_output, 'r') as file:
        data = json.load(file)

    entry_name = data.get("testing_entry_name", None)
    testing_entry = experiment_entry.get_kernel().byname(entry_name)
    return testing_entry


def lay_out(experiment_entries, division, submitter, log_truncation_script_path, submission_checker_path, sut_path, compliance_path, scenarios, power=False, model_meta_data=None, submitted_tree_path=None, model_mapping_path=None, __entry__=None):

    submitter_path      = make_local_dir( [ division, submitter ], submitted_tree_path)
    code_path           = make_local_dir( [ division, submitter, 'code'], submitted_tree_path)
    systems_path        = make_local_dir( [ division, submitter, 'systems'], submitted_tree_path )

    sut_descriptions_dictionary      = {}
    experiment_cmd_list = []

    if division=="open" and model_mapping_path:
        dst_file_path = os.path.join(submitter_path, "model_mapping.json")
        print(f"    Copying: {model_mapping_path}  -->  {dst_file_path}", file=sys.stderr)
        shutil.copy( model_mapping_path, dst_file_path)

    copy_readmes_for_code( experiment_entries, division, submitter, submitted_tree_path, power, __entry__ )

    generate_readmes_for_measurements( experiment_entries, division, submitter, submitted_tree_path, power, __entry__ )
    
    for experiment_entry in experiment_entries:

        target_scenario = experiment_entry['loadgen_scenario']
        scenario = target_scenario.lower()

        experiment_parameters = []

        if "power_loadgen_output" in experiment_entry["tags"]:
            power_experiment_entry = experiment_entry
            experiment_entry = get_testing_entry(experiment_entry)

        experiment_program_name  = experiment_entry.get('program_name')
        program_entry = __entry__.get_kernel().byname(experiment_program_name)

        src_dir = experiment_entry.get_path("")

        sut_name        = experiment_entry.get('sut_name')
        sut_description = experiment_entry.get('sut_description')
        loadgen_mode    = experiment_entry.get('loadgen_mode')


        compliance_test_name       = experiment_entry.get('loadgen_compliance_test')

        task = experiment_entry.get('task')
        display_benchmark   = task.replace("_", " ").title()

        mode = loadgen_mode.replace("Only", "")

        with_power      = experiment_entry.get("with_power")

        sut_descriptions_dictionary[sut_name] = sut_description

        mlperf_model_name = experiment_entry.get('mlperf_model_name')

        modified_program_name = experiment_program_name.replace("resnet50", "image_classification") 

        # ----------------------------[ measurements ]------------------------------------
        measurement_general_path = make_local_dir ( [ division, submitter, 'measurements', sut_name ], submitted_tree_path )
        measurement_path = make_local_dir( [ division, submitter, 'measurements', sut_name, mlperf_model_name, scenario], submitted_tree_path )

        for src_file_name in ( 'mlperf.conf', 'user.conf' ):
            src_file_path = os.path.join(src_dir, src_file_name)
            dst_file_path = os.path.join(measurement_path, src_file_name)
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
            raise RuntimeError(f"Key {e} is missing from model_meta_data or the model")

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
            results_path_syll   = [ division, submitter, 'results', sut_name, mlperf_model_name, scenario, mode]
        elif compliance_test_name  in ( "TEST01", "TEST04", "TEST06" ):
            results_path_syll = [ division, submitter, 'compliance', sut_name , mlperf_model_name, scenario , compliance_test_name ]
            if compliance_test_name in ( "TEST01", "TEST06" ):
                results_path_syll_TEST_acc = [ division, submitter, 'compliance', sut_name , mlperf_model_name, scenario , compliance_test_name, 'accuracy' ]
                results_path_TEST_acc = make_local_dir(results_path_syll_TEST_acc, submitted_tree_path)

        files_to_copy       = [ 'mlperf_log_summary.txt', 'mlperf_log_detail.txt' ]

        if mode=='accuracy' or compliance_test_name == "TEST01":
            files_to_copy.append( 'mlperf_log_accuracy.json' )
        if mode=='performance' and not compliance_test_name:
            if not with_power:
                results_path_syll.append( 'run_1' )

        if mode=='performance' and compliance_test_name in ( "TEST01", "TEST04", "TEST06" ):
            results_path_syll.extend(( mode, 'run_1' ))

        results_path = make_local_dir( results_path_syll, submitted_tree_path )

        for filename in files_to_copy:
            src_file_path = os.path.join(src_dir, filename)

            if (compliance_test_name in ( "TEST01", "TEST06" )  and filename == 'mlperf_log_accuracy.json'):
                dst_file_path = os.path.join(results_path_TEST_acc, filename)
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

                results_path = make_local_dir( results_path_syll, submitted_tree_path )

                for file_name in os.listdir(src_file_path):
                    if file_name != "ptd_out.txt":
                        src_file_path_file = src_file_path + file_name
                        results_path_file = results_path + "/" + file_name
                        shutil.copy(src_file_path_file, results_path_file)
                results_path_syll.remove(elem)

        if mode=='accuracy' or compliance_test_name == "TEST01":
            accuracy_content    = experiment_entry["accuracy_report"]
            if type(accuracy_content)==list:
                accuracy_content    = "\n".join( accuracy_content )
            elif type(accuracy_content)!=str:
                accuracy_content    = str( accuracy_content )

            if mode == 'accuracy':
                dst_file_path       = os.path.join(results_path, "accuracy.txt")
            elif compliance_test_name == "TEST01":
                dst_file_path       = os.path.join(results_path_TEST_acc, "accuracy.txt")

            with open(dst_file_path, "w") as fd:
                if mode=='accuracy':
                    print(f"    Storing accuracy -->  {dst_file_path}", file=sys.stderr)
                    fd.write(accuracy_content)
                    fd.write("\n")
                else:
                    print(f"    Creating empty file -->  {dst_file_path}", file=sys.stderr)

        if mlperf_model_name == 'stable-diffusion-xl' and mode=='accuracy':
            src_images_dir = os.path.join(src_dir, "images")
            results_images_path = os.path.join(results_path, "images")
            print(f"    Copying: {src_images_dir}  -->  {results_images_path}", file=sys.stderr)
            shutil.copytree( src_images_dir, results_images_path, dirs_exist_ok=True)

        # -------------------------------[ compliance , verification ]--------------------------------------
        if compliance_test_name in ( "TEST01", "TEST04", "TEST06" ):
            compliance_path_test = make_local_dir( [ division, submitter, 'compliance', sut_name , mlperf_model_name, scenario, compliance_test_name ], submitted_tree_path )

            ("Verification for ", compliance_test_name)

            tmp_dir = make_local_dir( [ division, submitter, 'compliance', sut_name , mlperf_model_name, scenario, 'tmp' ], submitted_tree_path )
            results_dir = os.path.join(submitter_path , 'results', sut_name, mlperf_model_name, scenario)
            compliance_dir = src_dir
            output_dir = os.path.join(submitter_path ,'compliance', sut_name , mlperf_model_name, scenario)
            verify_script_path =  os.path.join(compliance_path,compliance_test_name, "run_verification.py")
            if task in ["llm", "llama2", "llama3_1", "moe"]:
                dtype = experiment_entry['benchmark_output_data_type']
            else:
                dtype = ""
            result_verify =  __entry__.call('get', 'run_verify', create_run_verification_input_dict(task, tmp_dir, verify_script_path, results_dir, compliance_dir, output_dir, target_scenario, dtype))
            if result_verify == "":
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                raise RuntimeError(f"[get run_verify] failed to execute: {result_verify}")

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
        raise RuntimeError(f"[get run_truncation_script] failed to execute, and returned {result_trucation}")

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

    return submitted_tree_path


def run_checker(submitted_tree_path, division, submitter, submission_checker_path, checker_log_path, __entry__):

    result_checker =  __entry__.call( 'get', 'run_checker_script', {
            "submission_checker_path": submission_checker_path,
            "submitted_tree_path": submitted_tree_path
             } )

    print(result_checker)
    logfile = open(checker_log_path, "w")
    logfile.write(result_checker)


def full_run(experiment_entries, division, submitter, log_truncation_script_path, submission_checker_path, checker_log_path, sut_path, compliance_path, scenarios, power=False, model_meta_data=None, submitted_tree_path=None,  model_mapping_path=None, __entry__=None):
    """First run lay_out() to build the submission tree, then run_checker() to check its integrity.

Usage examples:

    # Here we use --substitution_map to infer Offline experiments from another mlperf_model_name while using "own" Server experiments.
    # In order to pacify submission_checker we pass in --model_mapping_path that maps all unrecognized "open" models onto "closed" ones known by the script.
    axs byname submitter , full_run  --framework=openai --task=llama2  --division=open --require_compliance+ --sut_name=xd670_h200_x8_sglang --program_name=llama2_using_openai_loadgen --sut_system_type=datacenter --submitter=Krai  --submission_entry_name=laid_out_sglang --extra_common_attributes,::=mlperf_model_name:llama3_1-70b-interactive-fp8_pre ---substitution_map='{"mlperf_model_name":{"llama3_1-70b-interactive-fp8_pre":"llama3_1-70b-fp8_pre"}}' --model_mapping_path=$HOME/work_collection/sglang_collection/model_mapping.json
    """

    if os.path.exists(submitted_tree_path):
        print("The path " + submitted_tree_path + " exists, skipping lay_out()")
    else:
        print("Run lay_out in {submitted_tree_path} ...")
        lay_out(experiment_entries, division, submitter, log_truncation_script_path, submission_checker_path, sut_path, compliance_path, scenarios, power, model_meta_data, submitted_tree_path, model_mapping_path, __entry__)

    print("Run checker...")
    run_checker(submitted_tree_path, division, submitter, submission_checker_path, checker_log_path, __entry__)


def generate_readmes_for_measurements(experiment_entries, division, submitter, submitted_tree_path, power, __entry__=None):
    
    readme_template_path = __entry__.get_path("README_template.md")

    for experiment_entry in experiment_entries:

        scenario        = experiment_entry['loadgen_scenario'].lower()

        src_dir         = experiment_entry.get_path("")
        sut_name        = experiment_entry.get('sut_name')
        
        loadgen_mode    = experiment_entry.get('loadgen_mode')
        with_power      = experiment_entry.get("with_power")
        
        target_qps = experiment_entry.get("loadgen_target_qps")
        target_latency = experiment_entry.get("loadgen_target_latency")

        experiment_cmd  = 'axs byquery ' + experiment_entry.get('__query')
        #experiment_cmd  = experiment_entry.get('produced_by')
        compliance_test_name      = experiment_entry.get('loadgen_compliance_test')
       
        # Use target_value only when the command is referred from "__query" tag
        if scenario in ['singlestream', 'multistream'] and "loadgen_target_latency" not in experiment_cmd:
            target_value = ",loadgen_target_latency=" + str(target_latency)
        elif scenario in ['offline', 'server'] and "loadgen_target_qps" not in experiment_cmd:
            target_value = ",loadgen_target_qps=" + str(target_qps)
        else:
            target_value = ""

        mlperf_round = 5.0 # FIXME: turn into a data_axs.json level parameter

        mode = loadgen_mode.replace("Only", "")

        print(f"Experiment: {experiment_entry.get_name()} living in {src_dir}\n  produced_by={experiment_cmd}\n     mode={mode}", file=sys.stderr)

        if "power_loadgen_output" in experiment_entry["tags"] and power:
            power_experiment_entry = experiment_entry
            avg_power = power_experiment_entry.call("avg_power")
            print(avg_power)
            experiment_entry = get_testing_entry(experiment_entry)
            power_case = True
        else:
            power_case = False
        
        mlperf_model_name = experiment_entry['mlperf_model_name']

        measurement_path = make_local_dir( [ division, submitter, 'measurements', sut_name, mlperf_model_name, scenario], submitted_tree_path )

        path_model_readme = os.path.join(measurement_path, "README.md")
        if os.path.exists(readme_template_path) and not os.path.exists(path_model_readme):
            with open(readme_template_path, "r") as input_fd:
                # Read the template
                template = input_fd.read()
            with open(path_model_readme, "w") as output_fd:
                # Write the formatted template to the target file
                output_fd.write( template.format( mlperf_round=mlperf_round, division=division.capitalize(), submitter=submitter, sut=sut_name,  model=mlperf_model_name, scenario=scenario ) )
        
        with open(path_model_readme, "a") as fd:
            if mode == 'Accuracy':
                fd.write( "\n" + "### Accuracy  " + "\n\n")
                fd.write( "```" + "\n" + experiment_cmd + "\n" + "```" + "\n\n")
            elif mode == 'Performance' and not compliance_test_name:
                if power_case:
                    fd.write( "### Power " + "\n\n")
                    fd.write( "```" + "\n" + experiment_cmd + "\n" + "```" + "\n\n")
                else:
                    fd.write( "### Performance " + "\n\n")
                    fd.write( "```" + "\n" + experiment_cmd + target_value + "\n" + "```" + "\n\n")
            else:
                fd.write( "### Compliance " + compliance_test_name + "\n\n")
                fd.write( "```" + "\n" + experiment_cmd + target_value + "\n" + "```" + "\n\n")
        print("")


def copy_readmes_for_code(experiment_entries, division, submitter, submitted_tree_path, power, __entry__):

    code_path = make_local_dir( [ division, submitter, 'code'], submitted_tree_path )

    sut_descriptions_dictionary      = {}

    for experiment_entry in experiment_entries:

        if power and "power_loadgen_output" in experiment_entry["tags"]:
            experiment_entry = get_testing_entry(experiment_entry)

        experiment_program_name  = experiment_entry.get('program_name')
        program_entry = __entry__.get_kernel().byname(experiment_program_name)
        mlperf_model_name = experiment_entry['mlperf_model_name']
        modified_program_name   = experiment_program_name.replace("resnet50", "image_classification")
        code_model_program_path = make_local_dir( [code_path, mlperf_model_name , modified_program_name ], submitted_tree_path )

        submission_files_to_copy_from_code = program_entry.get( "submission_files_to_copy_from_code" , [ "README.md" ] )
        for file_to_copy in submission_files_to_copy_from_code:
            file_to_copy_source_path = program_entry.get_path( file_to_copy )

            if os.path.exists(file_to_copy_source_path):
                print(f"    Copying: {file_to_copy_source_path}  -->  {code_model_program_path}", file=sys.stderr)
                shutil.copy(file_to_copy_source_path, code_model_program_path)
            else:
                print(f"    NOT Copying: {file_to_copy_source_path}  -->  {code_model_program_path}", file=sys.stderr)


def generate_table(experiment_entries, division, submitter, power, __entry__):

    col_names = ["SUT", "Scenario", "Mode / Compliance?", "Status", "Target metric", "Actual metric", "Power", "Efficiency"]
    table_data = []
    for experiment_entry in experiment_entries:
        
        scenario = experiment_entry['loadgen_scenario']

        entry_path = experiment_entry.get_path("")
        if power:
            power_experiment_path = os.path.join(entry_path, 'power_logs')
            performance_dir_path = os.path.join(power_experiment_path, 'run_1')
            mlperf_log_path = os.path.join(performance_dir_path, 'mlperf_log_summary.txt')
        elif not power:
            mlperf_log_path = os.path.join(entry_path, 'mlperf_log_summary.txt')
        sut_name = experiment_entry.get('sut_name')
        model_name = experiment_entry.get('model_name')
        loadgen_mode = experiment_entry.get('loadgen_mode')
        mode = loadgen_mode.replace("Only", "")
        target_qps = experiment_entry.get("loadgen_target_qps")
        target_latency = experiment_entry.get("loadgen_target_latency")
        compliance_test_name = experiment_entry.get('loadgen_compliance_test')
        if mode == "Accuracy":
            accuracy_metric = experiment_entry.get("accuracy_report")
        
        # Function to extract the actual performance metric
        def get_samples_per_second(file_path):
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        # Check for mlperf log
                        if "Scheduled samples per second" in line or "Samples per second" in line or "90th percentile latency (ns)" in line or "99th percentile latency (ns)" in line:
                            parts = line.split(':')
                            if len(parts) == 2:
                                performance = parts[1].strip()# Remove any leading/trailing whitespace
                                return performance  
            
            except IOError as e:
                print(f"Error reading file: {e}")
                return None

        #Function to extract the Result status
        def get_result_status(file_path):
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        # Check for VALID/INVALID result status
                        if "Result is" in line:
                            parts = line.split(':')
                            value = parts[1].strip()
                            return str(value)
            except IOError as e:
                print(f"Error reading file: {e}")
                return None

        # Function to extract accuracy for image-classification
        def extract_accuracy_ic(accuracy_metric):
            if accuracy_metric is not None and "accuracy=" in accuracy_metric:
                accuracy_part = accuracy_metric.split('accuracy=')[1]
                # Extracting the accuracy value before the '%' character
                accuracy_value = accuracy_part.split('%')[0].strip()
                return accuracy_value
            return "Accuracy value not found."

        # Function to extract accuracy for bert-99 and bert-99.9
        def extract_accuracy_bert(accuracy_metric):
            if accuracy_metric is not None and "\"f1\"" in accuracy_metric:
                accuracy_part = accuracy_metric.split(' \"f1\":')[1]
                accuracy_value = accuracy_part.split('}')[0].strip()
                return float(accuracy_value)
            return "F1 value not found."

        # Function to extract mAP value
        def extract_map(accuracy_metric):
            if accuracy_metric is not None:
                for item in accuracy_metric:
                    if "mAP=" in item:
                        # Extracting the numerical part of the mAP value
                        map_part = item.split('=')[1].strip()
                        map_value = map_part.split('%')[0].strip()
                        return map_value
            return "mAP value not found"

        def extract_accuracy_sdxl(accuracy_metric):
            if accuracy_metric is not None and "\'FID_SCORE\'" in accuracy_metric and "\'CLIP_SCORE\'" in accuracy_metric:
                fid_score_part = accuracy_metric.split('\'FID_SCORE\':')[1]
                fid_score_value = fid_score_part.split(',')[0].strip()

                clip_score_part = accuracy_metric.split('\'CLIP_SCORE\':')[1]
                clip_score_value = clip_score_part.split('}')[0].strip()

                return float(fid_score_value), float(clip_score_value)
            return "Scores not found."

        
        if power and "power_loadgen_output" in experiment_entry["tags"]:
            target_entry = get_testing_entry(experiment_entry)

            if scenario in ["Offline", "Server"]:
                target = target_entry.get('loadgen_target_qps')
            else:
                target = target_entry.get('loadgen_target_latency')

        # Target accuracy for workloads
        target_accuracy = {
            "resnet50": round(76.46 * 0.99, 3),
            "retinanet": round(37.55 * 0.99, 3),
            "bert-99": round(90.874 * 0.99, 3),
            "bert-99.9": round(90.874 * 0.999, 3),
            "stable-diffusion-xl": ("FID_SCORE", 23.01085758, "CLIP_SCORE", 31.68631873)
        }

        # Actual accuracy for workloads
        actual_accuracy = {
            "resnet50": extract_accuracy_ic(accuracy_metric),
            "retinanet": extract_map(accuracy_metric),
            "bert-99": extract_accuracy_bert(accuracy_metric),
            "bert-99.9": extract_accuracy_bert(accuracy_metric),
            "stable-diffusion-xl": extract_accuracy_sdxl(accuracy_metric)
        }

        # Accuracy upper limit
        accuracy_upper_limit = {
            "stable-diffusion-xl": ("FID_SCORE", 23.95007626, "CLIP_SCORE", 31.81331801)
        }

        target_acc = target_accuracy[model_name]
        actual_acc = actual_accuracy[model_name]

        if "power_loadgen_output" in experiment_entry["tags"] and power:
            power_experiment_entry = experiment_entry
            avg_power = round(power_experiment_entry.call("avg_power"),3)
        else:
            avg_power = "N/A"
        

        if mode.lower() == "performance":
            if scenario in ["Offline" , "Server"]:
                actual_metric = get_samples_per_second(mlperf_log_path)
                energy_eff = round(float(actual_metric)/float(avg_power) ,3) if power is True else "N/A"
            elif scenario in ["SingleStream", "MultiStream"]:
                actual_metric = float(get_samples_per_second(mlperf_log_path)) * 1e-6
                energy_eff = round(float(avg_power)/(1/float(actual_metric * 0.001)) ,3) if power is True else "N/A" # convert latency from milliseconds to seconds
            status = get_result_status(mlperf_log_path)

        else:
            if model_name == "stable-diffusion-xl":
                target_fid_score = target_acc[1]
                target_clip_score = target_acc[3]
                upper_fid_score = accuracy_upper_limit[model_name][1]
                upper_clip_score = accuracy_upper_limit[model_name][3]
                # Extract actual values
                if isinstance(actual_acc, tuple) and len(actual_acc) == 2:
                    actual_fid_score, actual_clip_score = actual_acc
                else:
                    raise ValueError("Invalid format for actual accuracy values")

                # Compare values within the range
                if target_fid_score <= actual_fid_score <= upper_fid_score and target_clip_score <= actual_clip_score <= upper_clip_score:
                    status = "VALID"
                else:
                    status = "INVALID"

                actual_metric =f"FID_SCORE: {actual_fid_score}\nCLIP_SCORE: {actual_clip_score}"
                target = f"FID_SCORE range: [{target_fid_score}, {upper_fid_score}]\nCLIP_SCORE range: [{target_clip_score}, {upper_clip_score}]"

            else:
                if (float(actual_acc) >= target_acc):
                    status = "VALID"
                else:
                    status = "INVALID"
                actual_metric = actual_acc
                target = target_acc
            energy_eff = "N/A"

        if scenario in ["Offline", "Server"] and mode.lower() == "performance" and not power:
            target = target_qps
        elif scenario in ["SingleStream", "MultiStream"] and mode.lower() == "performance" and not power:
            target = target_latency

        if compliance_test_name is False:
            mode = mode 
        elif compliance_test_name is None:
            mode = mode + ""
        else:
            mode = mode + " / " + compliance_test_name

        table_data.append([sut_name, scenario, mode, status, target, actual_metric, avg_power, energy_eff])

    # Display Table
    print(tabulate(table_data, headers=col_names, tablefmt="fancy_grid", stralign='center', floatfmt=".3f"))
