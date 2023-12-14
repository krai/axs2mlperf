import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

G_GPTJ6B_MAX_INPUT_SEQLEN = 1919

G_PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)

def prepare_tokenizer(checkpoint_path, padding_side="left"):

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=G_GPTJ6B_MAX_INPUT_SEQLEN,
        padding_side=padding_side,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def preprocess_cnndm_prompt(cnn_val_json_path):
    # Load from CNN dailymail
    with open(cnn_val_json_path, 'r') as fh:
        list_data_dict = json.load(fh)

    sources = [G_PROMPT_INPUT.format_map(
        example) for example in list_data_dict]
    targets = [f"{example['output']}" for example in list_data_dict]

    return sources, targets

def preprocess_files(source_dir,
                     destination_dir,
                     input_data_type,
                     new_file_extension,
                     tokenizer_path):

    tokenizer = prepare_tokenizer(tokenizer_path, padding_side="right")
    sources, targets = preprocess_cnndm_prompt(source_dir)

    input_batch = tokenizer.batch_encode_plus(
        sources, return_tensors="pt",
        padding='max_length', truncation=True,
        max_length=G_GPTJ6B_MAX_INPUT_SEQLEN
    )
    dtype = np.dtype(input_data_type)
    input_ids = input_batch.input_ids.numpy().astype(dtype)
    attention_mask = input_batch.attention_mask.numpy().astype(dtype)
    masked_tokens = 1 - attention_mask
    input_real_seqlen = np.sum(attention_mask, axis=1).astype(dtype)

    # np.save(os.path.join(destination_dir, f"input_ids_padded.{new_file_extension}"), input_ids)
    # np.save(os.path.join(destination_dir, f"attention_mask.{new_file_extension}"), attention_mask)
    # np.save(os.path.join(destination_dir, f"masked_tokens.{new_file_extension}"), masked_tokens)
    # np.save(os.path.join(destination_dir, f"input_lengths.{new_file_extension}"), input_real_seqlen)

    input_ids.tofile(os.path.join(destination_dir, f"input_ids_padded.{new_file_extension}"))
    attention_mask.tofile(os.path.join(destination_dir, f"attention_mask.{new_file_extension}"))
    masked_tokens.tofile(os.path.join(destination_dir, f"masked_tokens.{new_file_extension}"))
    input_real_seqlen.tofile(os.path.join(destination_dir, f"input_lengths.{new_file_extension}"))

    return

def preprocess(source_dir, input_data_type, new_file_extension, file_name, model_name_or_path, dataset_name, tags=None, entry_name=None, __record_entry__=None):
    
    __record_entry__["tags"] = tags or [ "preprocessed" ]
    entry_name_list = [ dataset_name, "preprocessed" ]
    entry_name = "_".join(entry_name_list)

    __record_entry__.save( entry_name )
    output_directory = __record_entry__.get_path(file_name)
    os.makedirs( output_directory )
    destination_dir = output_directory
    preprocess_files(source_dir, destination_dir, input_data_type, new_file_extension, model_name_or_path)
    return __record_entry__
