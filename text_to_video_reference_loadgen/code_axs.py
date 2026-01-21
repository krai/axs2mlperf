def get_volumes(volumes_mapping, output_entry=None):
    volume_strings = []
    for i in range(0, len(volumes_mapping), 2):
        host_path_key = volumes_mapping[i]
        container_path_key = volumes_mapping[i+1]
        host_path = output_entry.get(host_path_key)
        container_path = output_entry.get(container_path_key)
        volume_strings.append(f"-v {host_path}:{container_path}")
    return " ".join(volume_strings)