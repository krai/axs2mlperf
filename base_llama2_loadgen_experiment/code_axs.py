import json

from transformers import AutoTokenizer


def get_accuracy_dict(accuracy_dict_full):
    accuracy_dict = {}
    for k in accuracy_dict_full.keys():
        if k in ["rouge1", "rouge2", "rougeL", "tokens_per_sample"]:
            accuracy_dict[k] = accuracy_dict_full[k]
    return accuracy_dict

def parse_tokens(
    tokenised_accuracy_log_path: str, output_log_path: str
):
    with open(tokenised_accuracy_log_path) as f:
        log = json.load(f)

    output_log = []
    for item in log:
        hex_str = item["data"]
        hex_tokens = [hex_str[i : i + 8] for i in range(0, len(hex_str), 8)]
        tokens = [
            int.from_bytes(bytes.fromhex(tok), byteorder="little") for tok in hex_tokens
        ]
        output_log.append(tokens)

    with open(output_log_path, "w") as f:
        json.dump(output_log, f, indent=2)
    return output_log_path

def detokenise(
    checkpoint_path: str, tokenised_accuracy_log_path: str, output_log_path: str
):
    tokeniser = AutoTokenizer.from_pretrained(checkpoint_path)

    with open(tokenised_accuracy_log_path) as f:
        log = json.load(f)

    output_log = []
    for item in log:
        hex_str = item["data"]
        hex_tokens = [hex_str[i : i + 8] for i in range(0, len(hex_str), 8)]
        tokens = [
            int.from_bytes(bytes.fromhex(tok), byteorder="little") for tok in hex_tokens
        ]
        output_log.append({
            "seq_id" : item["seq_id"],
            "qsl_idx" : item["qsl_idx"],
            "data": tokeniser.decode(tokens),
            "token_count" : item["token_count"]
        })

    with open(output_log_path, "w") as f:
        json.dump(output_log, f, indent=2)
    return output_log_path

def extract_result(ignore_invalid, keep_prefixes, performance, iteration=-1, loadgen_scenario="UNSET", output_entry=None):
# Parsing the entry and extracting results.
    extracted_result = {}

    # Checking if the experiment is valid.
    # It will be invalid if at least one request was not delivered to a server.
    try:
        program_output_path = output_entry.get_path()
        with open(program_output_path) as f:
            output_parameters = json.load(f)
        experiment_valid = output_parameters["result_valid"]
    except:
        # Something went wrong with the experiment.
        experiment_valid = False

    extracted_result["Quality"] = -100.0

    if loadgen_scenario == "Server":
        ignore_invalid = True

    if experiment_valid:
        result_valid = True
        # Extracting and filtering the report, building a dictionary.
        for item in performance:
            if "=" in item:
                key, value = item.split("=")
                if key in keep_prefixes:
                    extracted_result[key] = value
            elif item == "INVALID" and not ignore_invalid:
                result_valid = False
        # Calculating the quality.
        if loadgen_scenario == "Offline":
            extracted_result["Quality"] = extracted_result["Samples_per_second"]
        elif loadgen_scenario == "Server":
            cutoff_ttft = float(extracted_result["cutoff_ratio_ttft"])
            cutoff_ttop = float(extracted_result["cutoff_ratio_tpot"])
            
            penalty = 0.0
            if cutoff_ttft > 1.0:
                penalty += min((cutoff_ttft - 1.0) * 5, 10)
            if cutoff_ttop > 1.0:
                penalty += min((cutoff_ttop - 1.0) * 5, 10)

            if penalty > 0:
                extracted_result["Quality"] = -penalty
            else:
                extracted_result["Quality"] = float(extracted_result["Completed_samples_per_second"])

        if not result_valid:
            extracted_result["Quality"] = -50.0

    # Adding the iteration number.
    extracted_result["Iteration"] = iteration

    return extracted_result
