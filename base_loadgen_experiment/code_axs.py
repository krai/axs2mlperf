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


def parse_performance(summary):

    if summary["Result_is"] == "INVALID":
        return None
    elif summary["Scenario"] == "Offline":
        return summary["Samples_per_second"]
    elif summary["Scenario"] == "SingleStream":
        return summary["_Early_stopping_90th_percentile_estimate"]
    elif summary["Scenario"] == "MultiStream":
        return summary["_Early_stopping_99th_percentile_estimate"]
    elif summary["Scenario"] == "Server":
        return summary["Scheduled_samples_per_second"]


def guess_command(tags, loadgen_scenario, loadgen_mode, model_name, loadgen_dataset_size, loadgen_buffer_size, loadgen_target_qps = None, loadgen_target_latency=None, loadgen_multistreamness=None ):

    terms_list = [] + tags
    terms_list.append( f"loadgen_scenario={loadgen_scenario}" )
    terms_list.append( f"loadgen_mode={loadgen_mode}" )
    terms_list.append( f"model_name={model_name}" )
    terms_list.append( f"loadgen_dataset_size={loadgen_dataset_size}" )
    terms_list.append( f"loadgen_buffer_size={loadgen_buffer_size}" )

    if loadgen_scenario == 'MultiStream':
        terms_list.append( f"loadgen_multistreamness={loadgen_multistreamness}" )

    if loadgen_mode == 'PerformanceOnly' and loadgen_scenario in ('SingleStream', 'MultiStream'):
        terms_list.append( f"loadgen_target_latency={loadgen_target_latency}" )

    if loadgen_mode == 'PerformanceOnly' and loadgen_scenario in ('Offline', 'Server'):
        terms_list.append( f"loadgen_target_qps={loadgen_target_qps}" )

    if loadgen_mode == 'AccuracyOnly' and loadgen_scenario == 'Server':
        terms_list.append( f"loadgen_target_qps={loadgen_target_qps}" )

    return "axs byquery "+','.join(terms_list)
