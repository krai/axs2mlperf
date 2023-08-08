#!/usr/bin/env python3

import datetime
import sys


def load_lines(fname):
    with open(fname) as f:
        return list(map(lambda x: x.rstrip("\n\r"), f))


def avg_power( mlperf_log_parser_path, server_timezone_sec, client_timezone_sec, detail_log_path, power_sample_log_path):

    sys.path.append( mlperf_log_parser_path )
    from log_parser import MLPerfLog

    server_timezone = datetime.timedelta(seconds=server_timezone_sec)
    client_timezone = datetime.timedelta(seconds=client_timezone_sec)

    mlperf_log = MLPerfLog(detail_log_path)

    datetime_format = '%m-%d-%Y %H:%M:%S.%f'
    power_begin = datetime.datetime.strptime(mlperf_log["power_begin"], datetime_format) + client_timezone
    power_end = datetime.datetime.strptime(mlperf_log["power_end"], datetime_format) + client_timezone

    power_list  = []
    for line in load_lines(power_sample_log_path):
        timestamp = datetime.datetime.strptime(line.split(",")[1], datetime_format) + server_timezone
        if timestamp > power_begin and timestamp < power_end:
            power_list.append(float(line.split(",")[3]))

    power_count = len(power_list)
    avg_power = sum(power_list) / power_count if power_count>0 else 'NO POWER DATA'

    return avg_power
