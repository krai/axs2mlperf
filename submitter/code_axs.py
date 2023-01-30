#!/usr/bin/env python3


import json
import os
import shutil
import subprocess
import sys

def store_json(data_structure, json_file_path):
    json_data   = json.dumps( data_structure , indent=4)

    with open(json_file_path, "w") as json_fd:
        json_fd.write( json_data+"\n" )


def lay_out(experiment_entries, division, submitter, record_entry_name, log_truncation_script_path, submission_checker_path, __entry__=None, __record_entry__=None):
    
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
                #__record_entry__.remove()
                return

        src_dir        = experiment_entry.get_path("")
        sut_name       = experiment_entry.get('sut_name')
        sut_data       = experiment_entry.get('sut_data')
        loadgen_mode   = experiment_entry.get('loadgen_mode')
        readme_path    = experiment_entry.get('program_entry').get_path("README.md")
        experiment_cmd = experiment_entry.get('produced_by')

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
        if os.path.exists(readme_template_path) and not os.path.exists(path_model_readme):
            shutil.copy(readme_template_path, path_model_readme)
            file = open(path_model_readme,"r").read()
            file = file.format(benchmark=benchmark, framework=framework )
            f = open(path_model_readme, 'w')
            f.write(file)
            f.close()
        file = open(path_model_readme,"a")
        file.write( "## Benchmarking " + model_name + " model " + "in " + mode + " mode" + "\n" + "```" + "\n" + experiment_cmd + "\n" + "```" + "\n\n")

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

        # NB: need to clear the cache somehow!  The attempt below did not work:
        experiment_entry.parent_objects = None

        # # -------------------------------[ results ]--------------------------------------
        mode        = {
            'AccuracyOnly': 'accuracy',
            'PerformanceOnly': 'performance',
         }[ experiment_entry['loadgen_mode'] ]
        results_path_syll   = ['submitted_tree', division, submitter, 'results', sut_name, display_model_name, scenario, mode]
        files_to_copy       = [ 'mlperf_log_summary.txt', 'mlperf_log_detail.txt' ]

        if mode=='accuracy':
            files_to_copy.append( 'mlperf_log_accuracy.json' )  # FIXME: 'accuracy.txt' has to be generated?
        elif mode=='performance':
            results_path_syll.append( 'run_1' )

        results_path        = make_local_dir( results_path_syll )

        for filename in files_to_copy:
            src_file_path = os.path.join(src_dir, filename)
            dst_file_path = os.path.join(results_path, filename)
            print(f"    Copying: {filename}  -->  {dst_file_path}", file=sys.stderr)
            shutil.copy( src_file_path, dst_file_path)

        if mode=='accuracy':
            if experiment_program_name == "object_detection_onnx_loadgen_py":
                accuracy_content    = str(experiment_entry["mAP"])
            elif experiment_program_name == "bert_squad_onnxruntime_loadgen_py":
                accuracy_content    = str(experiment_entry["f1"])
            elif experiment_program_name == "image_classification_onnx_loadgen_py" or experiment_program_name == "image_classification_torch_loadgen_py":
                accuracy_content    = str(experiment_entry["accuracy"])

            dst_file_path       = os.path.join(results_path, "accuracy.txt")
            print(f"    Storing accuracy -->  {dst_file_path}", file=sys.stderr)
            with open(dst_file_path, "w") as fd:
                 fd.write(accuracy_content)

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
