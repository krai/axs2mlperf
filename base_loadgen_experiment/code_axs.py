#!/usr/bin/env python3

from function_access import to_num_or_not_to_num

def parse_summary(abs_log_summary_path):

    parsed_summary = {}
    with open( abs_log_summary_path ) as log_summary_fd:
        for line in log_summary_fd:
            if ':' in line:     # for now, ignore all other lines
                k, v = (x.strip() for x in line.split(':', 1))
                k = k.replace(' ', '_').replace('/', '_').replace('*', '').replace(')', '').replace('(', '')

                parsed_summary[k] = to_num_or_not_to_num(v)
    return parsed_summary


def parse_performance(summary, scenario_performance_map, raw=False):

    scenario = summary["Scenario"]
    validity = summary["Result_is"]

    if raw and validity == "INVALID":
        return None

    key_name, multiplier, formatting, units = scenario_performance_map[scenario][validity]
    if raw:
        return summary[key_name]
    else:
        formatted_value = ('{:'+formatting+'}').format(summary[key_name]*multiplier)
        display_key_name = key_name.replace('_ns', '')
        return '{} : {}={}{}'.format(validity, display_key_name, formatted_value, units)


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


def guess_command(tags, framework, loadgen_scenario, loadgen_mode, model_name, loadgen_dataset_size, loadgen_buffer_size, loadgen_compiance_test = None, loadgen_target_qps = None, loadgen_target_latency=None, loadgen_multistreamness=None, sut_name = None):

    terms_list = [] + tags
    terms_list.append( f"framework={framework}" )
    terms_list.append( f"loadgen_scenario={loadgen_scenario}" )
    terms_list.append( f"loadgen_mode={loadgen_mode}" )
    terms_list.append( f"model_name={model_name}" )
    terms_list.append( f"loadgen_dataset_size={loadgen_dataset_size}" )
    terms_list.append( f"loadgen_buffer_size={loadgen_buffer_size}" )
    if loadgen_compiance_test is None:
        terms_list.append( f"loadgen_compiance_test-" )
    else:
        terms_list.append( f"loadgen_compiance_test={loadgen_compiance_test}" )
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
