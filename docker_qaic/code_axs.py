import getpass

def create_mounted_string(local_dir, container_local_dir, repo_list, local_experiment_dir_name, local_experiment_dir):
    result = ""
    for elem in repo_list:
        if elem not in [ "kilt-mlperf-dev_main", local_experiment_dir_name] :
            end = "_main"
        else:
            end = ""
        if elem == local_experiment_dir_name:
            local_dir = local_experiment_dir
        result += " -v " + local_dir + elem + ":" + container_local_dir + elem + end
    return result

def get_image_name(task):
    if task == "image_classification":
        image_name_part = "resnet50.full"
    elif task == "object_detection":
        image_name_part = "retinanet"
    elif task == "bert":
        image_name_part = "bert"
    return "krai/mlperf." + image_name_part

def get_model_name_container(model_name):
    if model_name in [ "bert-99", "bert-99.9" ]:
       result = "bert"
    else:
        result = model_name
    return result

def get_user_name():
    return getpass.getuser()
