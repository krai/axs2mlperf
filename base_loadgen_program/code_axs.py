import os
from shutil import copy2

def get_mlperf_model_name(model_name_compliance_dict, model_name):
    if model_name in model_name_compliance_dict.keys():
        print("DEBUG: model_name_dict[model_name] = ", model_name_compliance_dict[model_name])
        return model_name_compliance_dict[model_name]
    else:
        return None

def generate_user_conf(loadgen_param_dictionary, model_name, loadgen_scenario, target_user_conf_path, loadgen_mlperf_path, target_audit_conf_path, loadgen_compliance_test, compliance_test_config, model_name_compliance_dict):
    param_to_conf_pair = {
        "loadgen_count_override_min":   ("min_query_count", 1),
        "loadgen_count_override_max":   ("max_query_count", 1),
        "loadgen_multistreamness":      ("samples_per_query", 1),
        "loadgen_max_query_count":      ("max_query_count", 1),
        "loadgen_buffer_size":          ("performance_sample_count_override", 1),
        "loadgen_samples_per_query":    ("samples_per_query", 1),
        "loadgen_target_latency":       ("target_latency", 1),
        "loadgen_target_qps":           ("target_qps", 1),
        "loadgen_max_duration_s":       ("max_duration_ms", 1000),
        "loadgen_offline_expected_qps": ("offline_expected_qps", 1),
        "loadgen_min_duration_s":       ( "min_duration_ms", 1000),
    }

    user_conf   = []
    for param_name in loadgen_param_dictionary.keys():
        if param_name in param_to_conf_pair:
            orig_value = loadgen_param_dictionary[param_name]
            if orig_value is not None:
                (config_category_name, multiplier) = param_to_conf_pair[param_name]
                new_value = orig_value if multiplier==1 else float(orig_value)*multiplier
                user_conf.append("{}.{}.{} = {}\n".format(model_name, loadgen_scenario, config_category_name, new_value))

    with open(target_user_conf_path, 'w') as user_conf_file:
         user_conf_file.writelines(user_conf)

    if loadgen_compliance_test:
        target_audit_conf_path = generate_audit_conf( model_name, loadgen_mlperf_path, target_audit_conf_path, loadgen_compliance_test, compliance_test_config, model_name_compliance_dict)

    return target_user_conf_path

def generate_audit_conf( model_name, loadgen_mlperf_path, target_audit_conf_path, loadgen_compliance_test, compliance_test_config, model_name_compliance_dict):

    # Copy 'audit.config' for compliance testing into the current directory.
    mlperf_model_name = get_mlperf_model_name(model_name_compliance_dict, model_name)
    if mlperf_model_name is not None:
        model_name = mlperf_model_name

    path_parts = [ loadgen_mlperf_path, 'compliance', 'nvidia', loadgen_compliance_test ]

    if loadgen_compliance_test in [ 'TEST01' ]:
        path_parts.append(model_name)

    path_parts.append(compliance_test_config)
    compliance_test_config_source_path = os.path.join(*path_parts)
    if os.path.exists(compliance_test_config_source_path):
        copy2(compliance_test_config_source_path, target_audit_conf_path)
    else:
        raise Exception("Error: Missing compliance config file: '{}'".format(compliance_test_config_source_path))

    return target_audit_conf_path
