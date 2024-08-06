import argparse
import os

import numpy as np
import pandas as pd

PAD_TOKEN = 2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, help="The path to the dataset")
    parser.add_argument("--output-path", type=str, help="The path to the output")

    args = parser.parse_args()

    df = pd.read_pickle(args.dataset_path)
    dtype = np.dtype("int32")

    input_ids = np.full((len(df), 2048), PAD_TOKEN)
    input_lens = np.full((len(df), 1), 0)

    for i, (toks, tok_len) in enumerate(zip(df["tok_input"], df["tok_input_len"])):
        input_ids[i][2048 - tok_len:] = toks
        input_lens[i] = tok_len
    
    input_ids.tofile(os.path.join(args.output_path, "input_ids_padded.bin"))
    input_lens.tofile(os.path.join(args.output_path, "input_lengths.bin"))

