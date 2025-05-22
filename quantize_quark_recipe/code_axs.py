def get_script_extra_params(script_extra_params_dict):
    script_extra_params_str = ""
    for key, value in script_extra_params_dict.items():
        if isinstance(value, bool):
            if value:
                script_extra_params_str += f" --{key}"
        else:
            script_extra_params_str += f" --{key} {value}"
    return script_extra_params_str