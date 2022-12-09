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
        #print("Result_is = ", summary["Result_is"])
        return None
    elif summary["Scenario"] == "Offline":
        return summary["Samples_per_second"]
    elif summary["Scenario"] == "SingleStream":
        #print(" SS = ", summary["Early_stopping_90th_percentile_estimate"])
        return summary["_Early_stopping_90th_percentile_estimate"]
    elif summary["Scenario"] == "MultiStream":
        return summary["_Early_stopping_99th_percentile_estimate"]
    elif summary["Scenario"] == "Server":
        return summary["Scheduled_samples_per_second"]
