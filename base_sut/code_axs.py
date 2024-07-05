import os
import json
import re
import copy
import shutil
import math

def merge(sut, model, model_config_entry, model_compiletime_device_model_entry, runtime_device_model_entry, loadgen_scenario, num_device,
          system=None, with_power=None, power_server_address=None, power_server_port=None, power_max_amps=None, power_max_volts=None,
          profile_compiletime_device_model_entry=None, override_runtime_config=None, gen_profile_config_only=False, 
          device_id=None, cpu_entry=None, system_entry=None, __entry__=None, __record_entry__=None):
    """Assemble a list of suitable configs based on the loadgen_scenario, the model and the device(s) in the SUT. 
       Providing the loadgen_scenario is a must. cpu and system information are optional.
       Update priority: device_model combo > system > cpu

Usage examples :
        axs byquery sut_config,sut=chai,model=bert-99,loadgen_scenario=SingleStream
    """

    loadgen_scenarios = ["SingleStream", "Offline", "Server", "MultiStream"]
    assert loadgen_scenario in loadgen_scenarios , f"\n\nERROR: Invalid loadgen_scenario. The specified scenerio is [{loadgen_scenario}], but it should be one of these: {loadgen_scenarios} \n\n"

    __record_entry__.plant("tags", ["sut_config", f"sut={sut}"] )
    __record_entry__.plant("device_id", device_id)

    """
    Compiletime Config for Profile
    """
    data_dict = {}
    if profile_compiletime_device_model_entry:
        try:
            data_dict.update(profile_compiletime_device_model_entry.get("scenario_independent_compiletime_profile_config"))
        except:
            print(f"WARNING: scenario_independent_compiletime_profile_config not defined for {model_config_entry.get_path()}, defaulting to scenario_independent_compiletime_model_config.")
            data_dict.update(profile_compiletime_device_model_entry.get("scenario_independent_compiletime_model_config"))

        # Add profiling thread to data_dict if exist
        pt = profile_compiletime_device_model_entry.slice("profiling_thread", safe=True)
        if pt["profiling_thread"]:
            data_dict.update(pt)
        else:
            print("WARNING: profiling_thread not set")

        # Add device id, always compile on 1 device only
        data_dict["device_id"] = "0"

        # Add data to entry
        config_compiletime_profile = dict(sorted(data_dict.items()))
        __record_entry__.plant("config_compiletime_profile", config_compiletime_profile)
    else:
        print(f"WARNING: config_compiletime_profile not generated, probably because this model doesn't need profile.")

    if gen_profile_config_only:
        assert profile_compiletime_device_model_entry , f"\n\n\nERROR: profile_compiletime_device_model_entry doesn't exist, but trying to gen_profile_config_only.\n\n\n"
        return __record_entry__.save(f"assembled_{sut}_{data_dict['device_id']}_for_{model}_{loadgen_scenario}")

    """
    Compiletime Config for Model
    """
    data_dict = {}

    if model_compiletime_device_model_entry:
        try:
            if cpu_entry and cpu_entry.get(loadgen_scenario):
                data_dict.update(cpu_entry.get(loadgen_scenario))
        except:
            print("WARNING: Some cpu info are missing for this model/loadgen_scenario combination.")

        data_dict.update(model_compiletime_device_model_entry.get("scenario_independent_compiletime_model_config"))
        data_dict.update(model_compiletime_device_model_entry.get(loadgen_scenario))

        # Add profiling thread to data_dict if exist
        pt = model_compiletime_device_model_entry.slice("profiling_thread", safe=True)
        if pt["profiling_thread"]:
            data_dict.update(pt)
        else:
            print("WARNING: profiling_thread not set")
        # Add device id, always compile on 1 device only
        data_dict["device_id"] = "0"

        # Check for error in configs
        assert "ERROR" not in data_dict , f"\n\nERROR: There are error(s) in config(s). \n\n{data_dict['ERROR']} \n\n."

        # Add data to entry
        config_compiletime_model = dict(sorted(data_dict.items()))
        __record_entry__.plant("config_compiletime_model", config_compiletime_model)
    else:
        print(f"WARNING: config_compiletime_model not generated, probably because this sut cannot compile its own model.")

    """
    Runtime Config
    """
    data_dict = {}
    data_dict.update(runtime_device_model_entry.get("scenario_independent_runtime_config"))
    scenario_dependent_runtime_config = runtime_device_model_entry.get(loadgen_scenario)
    if scenario_dependent_runtime_config is None:
        raise ValueError(f"Expected runtime entry to have config related to {loadgen_scenario}")
    data_dict.update(scenario_dependent_runtime_config)

    try:
        data_dict.update(override_runtime_config[model][loadgen_scenario])
        print(f"WARNING: Overriding runtime config with override_runtime_config in {__entry__.get_path()}")
    except:
        pass

    # Add power
    if with_power:
        if power_server_address and power_server_port:
            data_dict["power_server_address"] = power_server_address
            data_dict["power_server_port"] = power_server_port
        if power_max_amps and power_max_volts:
            data_dict["power_max_amps"] = power_max_amps
            data_dict["power_max_volts"] = power_max_volts

        assert "power_server_address" in data_dict and "power_server_port" in data_dict, f"\n\n\nERROR: This sut {sut} has the setting with_power={with_power}, but we are missing power_server_address and power_server_port info. Please set them! \n\n"

    # Add device id
    if "device_id" in data_dict:
        print(f"WARNING: Found device_id={data_dict['device_id']} given in runtime config {runtime_device_model_entry.get_path()} [{loadgen_scenario}], will try to parse that [device_id={data_dict['device_id']}] instead of the default one [device_id={device_id}]...")
        device_id = data_dict["device_id"]

    final_device_id = _parse_device_id(num_device, text=device_id)
    if loadgen_scenario == "SingleStream" and len(final_device_id) > 1 and model != "stable-diffusion-xl":
        final_device_id = final_device_id[0]
        print(f"WARNING: Only use the first given valid device for SingleStream. Defaulting device_id to {final_device_id}.")
    data_dict["device_id"] = final_device_id

    # Add data to entry
    config_runtime = dict(sorted(data_dict.items()))
    __record_entry__.plant("config_runtime", config_runtime)

    """
    Inherit fan from system.
    """
    set_system_parent = False
    if system:
        system_entry = __entry__.get_kernel().byquery(f"system_config,system={system}")
        if system_entry:
            set_system_parent = True
            __record_entry__.plant("_parent_entries", [system_entry])
    if not set_system_parent:
        system_entry = __entry__.get_kernel().byname("base_system")
        __record_entry__.plant("_parent_entries", [system_entry])
        print(f"WARNING: Unable to find system=[{system}] in axs2system. Assume it will be base_system. Will not be able to inherit fan functionality.")

    return __record_entry__.save(f"assembled_{sut}_{data_dict['device_id']}_for_{model}_{loadgen_scenario}")

def gen_description(sut, num_device, system_type, with_power, __entry__=None, __record_entry__ = None):
    """Generate system description for MLPerf submissions.

Usage examples :
        axs byquery sut_description,sut=chai,model=bert-99
    """

    data_dict = {}

    base_sut_path = __entry__.get_kernel().byname("base_sut").get_path()
    sut_path = __entry__.get_path()

    template_description_path = os.path.join(base_sut_path, "description_template.json")
    template_power_setting_path =  os.path.join(base_sut_path, "power_settings_template.md" )
    template_analyzer_table_path = os.path.join(base_sut_path, "analyzer_table_template.md" )
    template_bios_setting_path = os.path.join(base_sut_path, "bios_setting_template.md" )

    if with_power:
        power_setting_path =  os.path.join(sut_path, "power_settings.md" )
        analyzer_table_path = os.path.join(sut_path, "analyzer_table.md" )
        assert os.path.isfile(power_setting_path) , f"\n\n\nERROR: Missing power_settings.md in {power_setting_path}, check out an example at {template_power_setting_path}\n"
        assert os.path.isfile(analyzer_table_path), f"\n\n\nERROR: Missing analyzer_table.md in {analyzer_table_path}, check out an example at {template_analyzer_table_path}\n"

    with open(template_description_path, 'r') as f:
        description = json.load(f)
        data_dict.update(description)

    data_dict["number_of_nodes"] = "1" # Num. of CPU
    data_dict["accelerators_per_node"] = str(num_device) # Num. of device
    data_dict["system_type"] = system_type

    description_path = os.path.join(__entry__.get_path(), "description.json")
    if os.path.isfile(description_path):
        with open(description_path, 'r') as f:
            description = json.load(f)
            data_dict.update(description)
    else:
        print(f"WARNING: {description_path} doesn't exist. Using the default template at {template_description_path}.")

    __record_entry__.plant("tags", ["sut_description", f"sut={sut}"] )
    __record_entry__.plant("data", dict(sorted(data_dict.items())) )
    __record_entry__.save(f"assembled_{sut}_description")
    output_entry_path = __record_entry__.get_path()

    if with_power:
        dest_power_setting_path = os.path.join(output_entry_path, "power_settings.md")
        dest_analyzer_table_path = os.path.join(output_entry_path, "analyzer_table.md")
        shutil.copy2(power_setting_path, dest_power_setting_path)
        shutil.copy2(analyzer_table_path, dest_analyzer_table_path)
    else:
        bios_setting_path = os.path.join(sut_path, "bios_setting.md" )
        if not os.path.isfile(bios_setting_path):
            print(f"WARNNING: Missing bios_setting.md in {bios_setting_path}, check out an example at {template_bios_setting_path }")
        else:
            dest_bios_setting_path = os.path.join(output_entry_path, "bios_setting.md")
            shutil.copy2(bios_setting_path, dest_bios_setting_path)

    return __record_entry__

def get_device_id(num_device):
    """Predict which devices the user want. E.g. if there are 3 cards, the func will return "0,1,2".
    """

    x = [str(i) for i in range(num_device)]
    print("x", x)
    return ",".join(x)

def _parse_device_id(num_device, text=None, delimiter="+" ):
    """Convert "d0d1d2" to "0,1,2"
    """
    print("XXX", text)
    print("XXX", text)
    print("XXX", text)
    
    original_text = text
    if isinstance(text, int):
        # Single number case
        assert text < num_device , f"\n\n\nError: Specified to use Card {text} but the System-Under-Test only has [{num_device}] cards.\n\n\n"
        return str(text)
    
    assert text and isinstance(text, str) , f"\n\n\nError: Expecting device_id in the format of '[digit]{delimiter}[digit]{delimiter}[digit]...' ,e.g., 0{delimiter}1{delimiter}3 for using card 0, card 1 and card 3, or device_id=all or a single number device_id=0, but we received [{original_text}].\n\n"

    if text == "all":
        return get_device_id(num_device)
    
    assert delimiter in text, f"\n\n\nError: Expecting device_id in the format of '[digit]{delimiter}[digit]{delimiter}[digit]...' ,e.g., 0{delimiter}1{delimiter}3 for using card 0, card 1 and card 3, or device_id=all or a single number device_id=0, but we received [{original_text}].\n\n"

    l = text.split(delimiter)
    text = ",".join(l)
    s = text.split(",")

    try:
        s = [eval(i) for i in s]
    except SyntaxError:
        raise Exception(f"\n\n\nError: Expecting device_id in the format of '[digit]{delimiter}[digit]{delimiter}[digit]...' ,e.g., 0{delimiter}1{delimiter}3 for using card 0, card 1 and card 3, or device_id=all or a single number device_id=0, but we received [{original_text}].\n\n")
    s = [int(x) for x in s]

    assert len(s) <= num_device and max(s) < num_device, f"\n\n\nError: Specified to use Card {s} but the System-Under-Test only has [{num_device}] cards.\n\n\n"

    return text

def set_hypothetical_num_device(num_device):
    """Sometimes we need a power of 2 number of cards, this function rounds down to them
    """
    
    MAX_SUPPORTED_NUM_DEVICE = 64
    assert num_device <= MAX_SUPPORTED_NUM_DEVICE, f"\n\n\nError: Supported up to {MAX_SUPPORTED_NUM_DEVICE} but the System-Under-Test has [{num_device}] cards.\n\n\n"

    return 2 ** int(math.log2(num_device))

def get_model_config(model, device, __entry__=None):
    query = f"model_config,model={model},device={device}"
    entry = __entry__.get_kernel().byquery(query)

    assert entry, f"\n\n\n ERROR: Should add model config for [{query}].\n\n"

    return entry

def get_config(config_target, model, device, card, system_type, num_device, hypothetical_num_device, __entry__=None):
    """Search for the relevant config. For compiletime parameters, use q1 version of it if the q_n version is not avalible. 

    """
    config_targets = ["compile_model", "compile_profile", "runtime"]
    assert config_target in config_targets, f"\n\nERROR: Invalid config_target. The specified config_target is [{config_target}], but it should be one of these: {config_targets} \n\n"

    if model in ["bert-99.9", "gptj-99", "gptj-99.9"] and config_target == "compile_profile":
        return None

    number_of_device = copy.deepcopy(num_device)
    if config_target != "runtime":
        if number_of_device != hypothetical_num_device:
            print(f"WARNING: In compile time, q{number_of_device} is similar to q{hypothetical_num_device}. Will be using those config instead.")
            number_of_device = hypothetical_num_device

    query = f"device_model_config,config_target={config_target},model={model},device={device},card={card},system_type={system_type},num_device={number_of_device}"
    latest_entry = __entry__.get_kernel().byquery(query)

    if latest_entry:
        print(f"Setting the {config_target} config from {latest_entry.get_path()}.")
        return latest_entry

    assert config_target != "runtime", f"\n\n\n ERROR: Should add runtime config for [{query}]."

    base_query = f"device_model_config,config_target={config_target},model={model},device={device},card={card},system_type={system_type},num_device=1"
    q1_entry = __entry__.get_kernel().byquery(base_query)

    if q1_entry:
        print(f"WARNING: Did not manage to find more up-to-date info. Setting the {config_target} config to be the same as the q1 version as defined in {q1_entry.get_path()}.")
        return q1_entry

    assert False, f"\n\n\nERROR: Missing Configs when searching for [{query}]. Try to add the relevant entry for the base class [{base_query}]."

