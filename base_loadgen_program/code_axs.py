import os
from shutil import copy2
from ufun import load_json, save_json


def generate_user_conf(loadgen_param_dictionary, shortened_mlperf_model_name, loadgen_scenario, target_user_conf_path, submission_compliance_tests_dir, target_audit_conf_path, loadgen_compliance_test, compliance_test_config):

    param_to_conf_pair = {
        "loadgen_count_override_min":   ("min_query_count", 1),
        "loadgen_count_override_max":   ("max_query_count", 1),
        "loadgen_multistreamness":      ("samples_per_query", 1),
        "loadgen_max_query_count":      ("max_query_count", 1),
        "loadgen_buffer_size":          ("performance_sample_count_override", 1),
        "loadgen_samples_per_query":    ("samples_per_query", 1),
        "loadgen_target_latency":       ("target_latency", 1),
        "loadgen_target_qps":           ("target_qps", 1),
        "loadgen_max_duration_s":       ("max_duration", 1000),
        "loadgen_min_duration_s":       ("min_duration", 1000),
        "loadgen_offline_expected_qps": ("offline_expected_qps", 1),
    }

    user_conf   = []
    for param_name in loadgen_param_dictionary.keys():
        if param_name in param_to_conf_pair:
            orig_value = loadgen_param_dictionary[param_name]
            if orig_value is not None:
                (config_category_name, multiplier) = param_to_conf_pair[param_name]
                if multiplier==1:
                    new_value = orig_value
                elif type(orig_value) is int and type(multiplier) is int:
                    new_value = int(orig_value)*multiplier
                else:
                    new_value = float(orig_value)*multiplier
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


def link_to_power_client_entry(output_entry, symlink_to, power_client_entrydic_path):
    "A callback procedure to be activated when in power measurement mode"

    power_workload_path = output_entry.get_path()
    power_workload_entry_name = output_entry.get_name()

    power_client_entry_path = os.path.dirname( symlink_to )

    if os.path.exists( symlink_to ):
        os.unlink( symlink_to )
        os.symlink( power_workload_path, os.path.join(power_client_entry_path, "testing_logs" ), target_is_directory=True )
        entrydic = load_json( power_client_entrydic_path )
        entrydic["testing_entry_name"] = power_workload_entry_name
    else:
        os.symlink( power_workload_path, os.path.join(power_client_entry_path, "ranging_logs" ), target_is_directory=True )
        entrydic = { "ranging_entry_name": power_workload_entry_name }

    save_json( entrydic, power_client_entrydic_path, indent=4 )

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
