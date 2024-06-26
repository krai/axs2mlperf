import os
from shutil import copy2
import datetime
from ufun import load_json, save_json


def generate_user_conf(loadgen_param_dictionary, param_to_conf_pair, shortened_mlperf_model_name, loadgen_scenario, target_user_conf_path, submission_compliance_tests_dir, target_audit_conf_path, loadgen_compliance_test, compliance_test_config):


    user_conf   = []
    for param_name in loadgen_param_dictionary.keys():
        orig_value = loadgen_param_dictionary[param_name]
        if orig_value is not None:
            (config_category_name, multiplier) = param_to_conf_pair[param_name]
            new_value = float(orig_value * multiplier)
            if float(int(new_value)) == new_value:
                new_value = int(new_value)
            user_conf.append("{}.{}.{} = {}\n".format(shortened_mlperf_model_name, loadgen_scenario, config_category_name, new_value))

    with open(target_user_conf_path, 'w') as user_conf_file:
         user_conf_file.writelines(user_conf)

    if loadgen_compliance_test:
        target_audit_conf_path = generate_audit_conf( shortened_mlperf_model_name, submission_compliance_tests_dir, target_audit_conf_path, loadgen_compliance_test, compliance_test_config )

    return target_user_conf_path


def generate_audit_conf( shortened_mlperf_model_name, submission_compliance_tests_dir, target_audit_conf_path, loadgen_compliance_test, compliance_test_config ):

    # Copy 'audit.config' for compliance testing into the current directory.
    path_parts = [ submission_compliance_tests_dir, loadgen_compliance_test ]

    if loadgen_compliance_test in [ 'TEST01' ]:
        path_parts.append(shortened_mlperf_model_name)

    path_parts.append(compliance_test_config)
    compliance_test_config_source_path = os.path.join(*path_parts)
    if os.path.exists(compliance_test_config_source_path):
        copy2(compliance_test_config_source_path, target_audit_conf_path)
    else:
        raise Exception("Error: Missing compliance config file: '{}'".format(compliance_test_config_source_path))

    return target_audit_conf_path


def link_to_power_client_entry(output_entry, symlink_to, power_client_entrydic_path, effective_no_ranging):
    "A callback procedure to be activated when in power measurement mode"

    power_workload_path = output_entry.get_path()
    power_workload_entry_name = output_entry.get_name()

    power_client_entry_path = os.path.dirname( symlink_to )

    if os.path.exists(power_client_entrydic_path ):
        entrydic = load_json( power_client_entrydic_path )

        logs_name = "testing_logs"
        entry_name_dict = "testing_entry_name"
    else:
        entrydic = {}
        if effective_no_ranging:
            logs_name = "testing_logs"
            entry_name_dict = "testing_entry_name"
        else:
            logs_name = "ranging_logs"
            entry_name_dict = "ranging_entry_name"

    entrydic[entry_name_dict] = power_workload_entry_name
    save_json( entrydic, power_client_entrydic_path, indent=4 )

    if os.path.exists( symlink_to ):
        os.unlink( symlink_to )
    os.symlink( power_workload_path, symlink_to, target_is_directory=True )

    return output_entry


def get_config_from_sut(config=None, default_val=None, sut_data_runtime=None, sut_data_compiletime=None, sut_entry=None):
    # Check if config is in sut_data_runtime
    if sut_data_runtime and config in sut_data_runtime:
        print(f"Setting {config} with {sut_entry.get_path()} from sut_data_runtime...")
        return sut_data_runtime[config]

    # If not in sut_data_runtime, check if config is in sut_data_compiletime
    elif sut_data_compiletime:
        if config in sut_data_compiletime:
            print(f"Setting {config} with {sut_entry.get_path()} from sut_data_compiletime...")
            return sut_data_compiletime[config]

        # Special condition for ml_model_seq_length
        elif config == "ml_model_seq_length" and "onnx_define_symbol" in sut_data_compiletime:
            if "seg_length" in sut_data_compiletime["onnx_define_symbol"]:
                return sut_data_compiletime["onnx_define_symbol"]["seg_length"]

    # If config is not in either, return default_val
    print(f"Bailing, set {config} to [{default_val}] ...")
    return default_val

def generate_current_timestamp(used_for="unknown"):
    timestamp = datetime.datetime.now().strftime("%Y.%m.%dT%H:%M:%S")
    #print(f"GENERATING TIMESTAMP: {timestamp} used for: {used_for}")
    return timestamp
