import pickle
from functools import partial

import pandas as pd
from transformers import AutoTokenizer

def format_input(row, sys_prompt_msg: str, no_sys_prompt_msg: str):
    if row["system_prompt"]:
        return sys_prompt_msg.format(row["system_prompt"], row["question"])
    else:
        return no_sys_prompt_msg.format(row["question"])

def _tokenize_helper(x, llama_tokenizer=None):
    if not isinstance(x, str):
        return []
    return llama_tokenizer(x)["input_ids"]


def convert_pickle_file_llama2_to_llama3(input_pkl_path: str, output_pkl_path: str, tokeniser_path: str, model_family: str):
    tok = AutoTokenizer.from_pretrained(tokeniser_path)
    with open(input_pkl_path, "rb") as f:
        df = pickle.load(f)
        
    model_family_to_msgs = {
        "llama3": {
            "sys": "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "no_sys": "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        },
        "llama3_1": {
            "sys": "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "no_sys": "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        },
        "qwen2_5": {
            "sys": "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "no_sys": "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        }
    }
    
    sys = model_family_to_msgs[model_family]["sys"]
    no_sys = model_family_to_msgs[model_family]["no_sys"]
        
    formatter = partial(format_input, sys_prompt_msg=sys, no_sys_prompt_msg=no_sys)

    df["input"] = df.apply(formatter, axis=1)

    input_tokenizer = partial(_tokenize_helper, llama_tokenizer=tok)
    output_tokenizer = partial(_tokenize_helper, llama_tokenizer=tok)
    df["tok_input"] = df["input"].apply(input_tokenizer)
    df["tok_output"] = df["output"].apply(output_tokenizer)
    df["tok_input_length"] = df["tok_input"].apply(lambda x: len(x))
    df["tok_output_length"] = df["tok_output"].apply(lambda x: len(x))

    print(df["input"][0])
    print(input_tokenizer(df["input"][0]))
    print(df["tok_input"][0] == input_tokenizer(df["input"][0]))

    with open(output_pkl_path, "wb") as f:
        pickle.dump(df, f)

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Converts a the openorca dataset from the llama2 format to the llama3 format")
    parser.add_argument("--input_pkl_path", required=True, type=str, help="Path to input pickle llama2 file")
    parser.add_argument("--output_pkl_path", required=True, type=str, help="Path to output pickle llama3 file")
    parser.add_argument("--tokeniser_path", required=True, type=str, help="Path to the tokeniser")
    parser.add_argument("--model_family", required=True, type=str, help="Model family to tokenise to")

    return parser.parse_args()

if __name__ == "__main__":
    convert_pickle_file_llama2_to_llama3(**vars(parse_args()))
