def get_volumes(volumes_mapping, output_entry=None):
    volume_strings = []
    for host_path_key, container_path_key in volumes_mapping:
        host_path = output_entry.get(host_path_key)
        container_path = output_entry.get(container_path_key)
        volume_strings.append(f"-v {host_path}:{container_path}")
    return " ".join(volume_strings)