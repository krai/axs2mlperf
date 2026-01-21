def get_volumes(volumes_mapping, output_entry=None):
    volume_strings = []
    for i in range(0, len(volumes_mapping), 2):
        host_path = volumes_mapping[i]
        container_path = volumes_mapping[i+1]
        volume_strings.append(f"-v {host_path}:{container_path}")
    return " ".join(volume_strings)