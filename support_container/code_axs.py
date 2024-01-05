def create_mounted_string(local_dir, container_local_dir, repo_list, local_experiment_dir_name):
    result = ""
    repo_list.append(local_experiment_dir_name)
    for elem in repo_list:
        if elem not in [ "kilt-mlperf-dev_main", local_experiment_dir_name] :
            end = "_main"
        else:
            end = ""
        result += " -v " + local_dir + elem + ":" + container_local_dir + elem + end
    #result += " -v " + local_dir + local_experiment_dir_name + ":" + container_local_dir + local_experiment_dir_name
    return result
    
