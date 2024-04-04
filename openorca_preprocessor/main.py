import sys
import os
import transformers
import pandas as pd
import numpy as np

args = iter(sys.argv[1:])

dataset_path = next(args)
tokenizer_path = next(args)
output_path = next(args)

df = pd.read_pickle(dataset_path)

tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

dtype = np.dtype("int32")

input_ids = np.full((len(df), 1024), tokenizer.eos_token_id)
attention_mask = np.full((len(df), 1024), 0)
for i, row in enumerate(df["tok_input"]):
    input_ids[i][:len(row)] = row
    attention_mask[i][:len(row)] = 1

input_ids = input_ids.astype(dtype)
attention_mask = attention_mask.astype(dtype)
masked_tokens = 1 - attention_mask
input_real_seqlen = np.sum(attention_mask, axis=1).astype(dtype)

input_ids.tofile(os.path.join(output_path, "input_ids_padded.bin"))
attention_mask.tofile(os.path.join(output_path, "attention_mask.bin"))
masked_tokens.tofile(os.path.join(output_path, "masked_tokens.bin"))
input_real_seqlen.tofile(os.path.join(output_path, "input_lengths.bin"))
