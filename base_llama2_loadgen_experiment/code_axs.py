import json

from transformers import AutoTokenizer


def get_accuracy_dict(accuracy_dict_full):
    accuracy_dict = {}
    for k in accuracy_dict_full.keys():
        if k in ["rouge1", "rouge2", "rougeL", "tokens_per_sample"]:
            accuracy_dict[k] = accuracy_dict_full[k]
    return accuracy_dict

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
            int.from_bytes(bytes.fromhex(tok), byteorder="big") for tok in hex_tokens
        ]
        output_log.append(tokeniser.decode(tokens))

    with open(output_log_path, "w") as f:
        json.dump(output_log, f, indent=2)
    return output_log_path
