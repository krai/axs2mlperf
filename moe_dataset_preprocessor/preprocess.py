import argparse
import json
import os

import numpy as np
import pandas as pd

PAD_TOKEN = 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, help="The path to the dataset")
    parser.add_argument("--output-path", type=str, help="The path to the output")
    parser.add_argument("--right-align", type=bool, default=False, help="Whether or not to right align the input")

    args = parser.parse_args()

    df = pd.read_pickle(args.dataset_path)
    dtype = np.dtype("int32")
    max_input_length = 2048 

    input_ids = np.full((len(df), max_input_length), PAD_TOKEN, dtype=dtype)
    input_lens = np.full((len(df), 1), 0, dtype=dtype)
    inputs = []

    for i, (toks, tok_len, text) in enumerate(zip(df["tok_input"], df["tok_input_len"], df["input"])):
        if args.right_align:
            input_ids[i][max_input_length - tok_len:] = toks
        else:
            input_ids[i][:tok_len] = toks
        input_lens[i] = tok_len
        inputs.append({"qsl_idx": i, "text": text})

    
    input_ids.tofile(os.path.join(args.output_path, "input_ids_padded.bin"))
    input_lens.tofile(os.path.join(args.output_path, "input_lengths.bin"))

    with open(os.path.join(args.output_path, "input_text.json"), "w") as f:
        json.dump(inputs, f, indent=2)

