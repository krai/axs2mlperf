import getpass
import socket

def get_user_name():
    return getpass.getuser()

def get_hostname():
    return socket.gethostname()

def create_mounted_string(local_dir, container_local_dir, repo_list, local_experiments_dir, local_experiment_dir_name, source_dir, code_dir):
    result = ""
    for elem in repo_list:
        if elem != local_experiment_dir_name:
            end = "_main"
        else:
            end = ""
        if elem == local_experiment_dir_name:
            local_dir = local_experiments_dir

        result += " -v " + local_dir + elem + ":" + container_local_dir + elem + end
    result += " -v " + source_dir + ":" + code_dir 
    return result

def create_ccache_dirs(code_dir, cache_dir):
    result = " --env \"CCACHE_DIR=" + code_dir + cache_dir + "\"" + " --env \"CCACHE_BASEDIR=" + code_dir + "\"" + " "
    return result
