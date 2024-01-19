#!/bin/sh

# First get the `open-orca` parquet from huggingface
export OPENORCA_DATASET=${PWD}/open-orca
git clone https://huggingface.co/datasets/Open-Orca/OpenOrca ${OPENORCA_DATASET}

export OPENORCA_PARQUET=${OPENORCA_DATASET}/1M-GPT4-Augmented.parquet
EXPORT_DIR=${PWD}/processed-openorca
export DATASET_PATH=${PWD}/processed-data.pkl

# Process the dataset according the Taskforce's agreed criteria
python3 processorca.py --dataset_pq_path=${OPENORCA_PARQUET} --model_dir=${CHECKPOINT_PATH} --seqlen_limit=1024 --export_dir=${EXPORT_DIR} --num_total_samples=24576

mv ${EXPORT_DIR}/open_orca_gpt4_tokenized_llama.sampled_24576.pkl ${DATASET_PATH}