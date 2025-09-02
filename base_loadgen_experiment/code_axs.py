#!/usr/bin/env python3

from function_access import to_num_or_not_to_num
from pint import UnitRegistry
import numpy as np      # needed for denumpify_scalar() and denumpify_dict()


def parse_summary(abs_log_summary_path):

    parsed_summary = {}
    with open( abs_log_summary_path ) as log_summary_fd:
        for line in log_summary_fd:
            if ':' in line:     # for now, ignore all other lines
                k, v = (x.strip() for x in line.split(':', 1))
                k = k.replace(' ', '_').replace('/', '_').replace('*', '').replace(')', '').replace('(', '')

                parsed_summary[k] = to_num_or_not_to_num(v)
    
    return parsed_summary


def beautify_summary(parsed_summary):

    ureg = UnitRegistry()

    linked_keys = ["latency", "Early_stopping_9", "duration", "time_to_output_token"]

    beautified_summary = {}
    kv_with_units = {}
    # Pretty print units
    for k, v in parsed_summary.items():
        k: str
        unit = None
        if k.endswith("_ns"):
            k = k[:-3]
            unit = ureg.ns
        elif k.endswith("_ms"):
            k = k[:-3]
            unit = ureg.ms
        elif k.startswith("_Early_stopping_9"):
             # Edge case for _Early_stopping_90th_percentile_estimate
             #           and _Early_stopping_99th_percentile_estimate
            unit = ureg.ns
        
        if unit is None:
            beautified_summary[k] = v
            continue
        
        v = (v*unit).to_compact()

        if v.u == ureg.us: 
            v.ito(ureg.ms) # Keep everything in milliseconds

        kv_with_units[k] = v

    for linked_key in linked_keys:
        linked = {k:v for k,v in kv_with_units.items() if linked_key in k}
        if len(linked) == 0:
            continue

        largest_unit = max(linked.values(), key=lambda v: v.u)
        for k, v in linked.items():
            v.ito(largest_unit)
            unit_suffix = f"{v.u}{'s' if v.m != 1 else ''}"
            beautified_summary[k] = f"{v.m :<.3f} {unit_suffix}"
    
    return beautified_summary

def calc_latency_cutoff_ratio(parsed_summary, latency, target_latency):

    scenario = parsed_summary["Scenario"]
    if scenario == "Server":
        if target_latency in parsed_summary:
            return parsed_summary[latency]/parsed_summary[target_latency]

def calc_early_stopping_overhead(parsed_summary):

    scenario = parsed_summary["Scenario"]

    if scenario == "SingleStream":
        return 100*(parsed_summary["_Early_stopping_90th_percentile_estimate"]/parsed_summary["90.00_percentile_latency_ns"]-1)
    elif scenario == "MultiStream":
        return 100*(parsed_summary["_Early_stopping_99th_percentile_estimate"]/parsed_summary["99.00_percentile_latency_ns"]-1)


#returns list of formatted performance metrics (as strings) for given experiment  
def parse_performance(parsed_summary, beautified_summary, early_stopping_overhead, scenario_performance_map, raw=False):

    scenario = beautified_summary["Scenario"]
    validity = beautified_summary["Result_is"]

    if raw and validity == "INVALID":
        return None  

    performance_metrics = scenario_performance_map[scenario]
    formatted_performance_metrics = ['{}'.format(validity)] # set first element 

    for key_name in performance_metrics:

        if key_name == "latency_cutoff_ratio":
            fmt, value = '{}={:.2f}', calc_latency_cutoff_ratio(parsed_summary, "99.00_percentile_latency_ns", "target_latency_ns")
        elif key_name == "cutoff_ratio_ttft":
            fmt, value = '{}={:.2f}', calc_latency_cutoff_ratio(parsed_summary, "99.00_percentile_first_token_latency_ns", "ttft_latency_ns")
        elif key_name == "cutoff_ratio_tpot":
            fmt, value = '{}={:.2f}', calc_latency_cutoff_ratio(parsed_summary, "99.00_percentile_time_to_output_token_ns", "tpot_latency_ns")
        elif key_name == "early_stopping_overhead":
            fmt, value = '{}={:.2f}%', early_stopping_overhead
        elif key_name in beautified_summary:
            fmt, value = '{}={}', beautified_summary[key_name]
        else:
            value = None

        if value is not None:
            if raw:
                formatted_performance_metrics.append( value )
            else:
                formatted_performance_metrics.append( fmt.format(key_name, value) )

    return formatted_performance_metrics


def unpack_accuracy_log(raw_accuracy_log):
    import struct
    def unpack_one_blob(packed_blob):
        return list(struct.unpack(f'{int(len(packed_blob)/8)}f', bytes.fromhex(packed_blob)))

    readable_accuracy_log = []
    for orig_record in raw_accuracy_log:
        readable_accuracy_log.append( {
            "seq_id": orig_record["seq_id"],
            "qsl_idx": orig_record["qsl_idx"],
            "data": unpack_one_blob(orig_record["data"]),
        } )
    return readable_accuracy_log


def guess_command(tags, framework, loadgen_scenario, loadgen_mode, model_name, loadgen_dataset_size, loadgen_buffer_size, loadgen_compliance_test = None, loadgen_target_qps = None, loadgen_target_latency=None, loadgen_multistreamness=None, sut_name = None):

    terms_list = [] + tags
    terms_list.append( f"framework={framework}" )
    terms_list.append( f"loadgen_scenario={loadgen_scenario}" )
    terms_list.append( f"loadgen_mode={loadgen_mode}" )
    terms_list.append( f"model_name={model_name}" )
    terms_list.append( f"loadgen_dataset_size={loadgen_dataset_size}" )
    terms_list.append( f"loadgen_buffer_size={loadgen_buffer_size}" )
    if loadgen_compliance_test is None:
        terms_list.append( "loadgen_compliance_test-" )
    else:
        terms_list.append( f"loadgen_compliance_test={loadgen_compliance_test}" )
    if sut_name is not None:
      terms_list.append( f"sut_name={sut_name}" )

    if loadgen_scenario == 'MultiStream':
        terms_list.append( f"loadgen_multistreamness={loadgen_multistreamness}" )

    if loadgen_mode == 'PerformanceOnly' and loadgen_scenario in ('SingleStream', 'MultiStream'):
        terms_list.append( f"loadgen_target_latency={loadgen_target_latency}" )

    if loadgen_mode == 'PerformanceOnly' and loadgen_scenario in ('Offline', 'Server'):
        terms_list.append( f"loadgen_target_qps={loadgen_target_qps}" )

    if loadgen_mode == 'AccuracyOnly' and loadgen_scenario == 'Server':
        terms_list.append( f"loadgen_target_qps={loadgen_target_qps}" )

    return "axs byquery "+','.join(terms_list)


def validate_accuracy(accuracy_dict, accuracy_range_dict):
    result_list = []
    for key in accuracy_dict:
        if key not in accuracy_range_dict:
            continue
        elif accuracy_range_dict[key][0] is not  None  and accuracy_range_dict[key][1] is not  None:
            if (accuracy_dict[key] >= accuracy_range_dict[key][0] and  accuracy_dict[key] <= accuracy_range_dict[key][1]):
                validity = "VALID"
            else:
                validity = "INVALID"
        elif accuracy_range_dict[key][0] is not  None  and accuracy_range_dict[key][1] is None:
            if accuracy_dict[key] >= accuracy_range_dict[key][0]:
                validity = "VALID"
            else:
                validity = "INVALID"
        elif accuracy_range_dict[key][0] is None  and accuracy_range_dict[key][1] is not None:
            if accuracy_dict[key] <= accuracy_range_dict[key][1]:
                validity = "VALID"
            else:
                validity = "INVALID"
        print('{} : {}={}'.format(validity, key, accuracy_dict[key]) )


def denumpify_scalar(scalar):
    "Transform individual values like np.int32(12345) or np.float32(12.3456) into Python's own 12345 and 12.3456"

    if hasattr(scalar, 'item'):
        return scalar.item()
    else:
        return scalar


def denumpify_dict(np_dict):
    "Transform a dictionary's values using denumpify_scalar()"

    if type(np_dict)==str:
        np_dict = eval(np_dict)

    return { k: denumpify_scalar(np_dict[k]) for k in np_dict }

def extract_result(ignore_invalid, keep_prefixes, performance,
                   iteration=-1, loadgen_scenario="UNSET", __entry__=None):
# Parsing the entry and extracting results.
    extracted_result = {}

    # Checking if the experiment is valid.
    # It will be invalid if at least one request was not delivered to a server.
    try:
        program_output_path = __entry__.get_path("program_output.json")
        with open(program_output_path) as f:
            output_parameters = json.load(f)
        experiment_valid = output_parameters["result_valid"]
    except:
        # Something went wrong with the experiment.
        experiment_valid = False

    extracted_result["Quality"] = -100.0

    if loadgen_scenario == "Server":
        ignore_invalid = True

    if experiment_valid:
        result_valid = True
        # Extracting and filtering the report, building a dictionary.
        for item in performance:
            if "=" in item:
                key, value = item.split("=")
                if key in keep_prefixes:
                    extracted_result[key] = value
            elif item == "INVALID" and not ignore_invalid:
                result_valid = False
        # Calculating the quality.
        if loadgen_scenario == "Offline":
            extracted_result["Quality"] = extracted_result["Samples_per_second"]
        elif loadgen_scenario == "Server":
            cutoff_ttft = float(extracted_result["cutoff_ratio_ttft"])
            cutoff_ttop = float(extracted_result["cutoff_ratio_tpot"])
            
            penalty = 0.0
            if cutoff_ttft > 1.0:
                penalty += min((cutoff_ttft - 1.0) * 5, 10)
            if cutoff_ttop > 1.0:
                penalty += min((cutoff_ttop - 1.0) * 5, 10)

            if penalty > 0:
                extracted_result["Quality"] = -penalty
            else:
                extracted_result["Quality"] = float(extracted_result["Completed_samples_per_second"])

        if not result_valid:
            extracted_result["Quality"] = -50.0

    # Adding the iteration number.
    extracted_result["Iteration"] = iteration

    return extracted_result
