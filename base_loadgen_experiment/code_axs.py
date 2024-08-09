#!/usr/bin/env python3

from function_access import to_num_or_not_to_num
from pint import UnitRegistry


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

    linked_keys = ["latency", "Early_Stopping_9", "duration"]

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
        elif "Early_Stopping_9" in k:
             # Edge case for _Early_Stopping_90th_percentile_estimate 
             #           and _Early_Stopping_99th_percentile_estimate
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


def calc_latency_cutoff_ratio(parsed_summary):
    
    scenario = parsed_summary["Scenario"]

    if scenario == "Server":
        return parsed_summary["99.00_percentile_latency_ns"]/parsed_summary["target_latency_ns"]


#returns list of formatted performance metrics (as strings) for given experiment  
def parse_performance(beautified_summary, latency_cutoff_ratio, scenario_performance_map, raw=False):

    scenario = beautified_summary["Scenario"]
    validity = beautified_summary["Result_is"]

    if raw and validity == "INVALID":
        return None  

    performance_metrics = scenario_performance_map[scenario][validity] 
    formatted_performance_metrics = ['{}'.format(validity)] # set first element 

    for key_name in performance_metrics:

        if raw:
            if key_name == "latency_cutoff_ratio":
                formatted_performance_metrics.append(latency_cutoff_ratio)
            else:
                formatted_performance_metrics.append(beautified_summary[key_name])
               
        else: #no need for multiplier, formatting, units in scenario_performance_map - the beautify_summary function does all of this already 
            if key_name == "latency_cutoff_ratio":
                formatted_performance_metrics.append('{}={}'.format(key_name, latency_cutoff_ratio))
            else:
                formatted_performance_metrics.append('{}={}'.format(key_name, beautified_summary[key_name]))

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
