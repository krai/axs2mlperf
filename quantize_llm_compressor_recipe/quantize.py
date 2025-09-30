from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor.transformers import SparseAutoModelForCausalLM
from llmcompressor import oneshot

import sys

args = iter(sys.argv[1:])

source_model_path = next(args)
target_model_path = next(args)
calib_dataset_path = next(args)
num_gpus = int(next(args))
max_sequence_length = int(next(args))
num_calibration_samples = int(next(args))
calibration_target = next(args)

recipe_fp8 = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
    """

recipe_fp4 = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 4
                        type: float
                        strategy: tensor_group
                        dynamic: false
                        symmetric: true
                        group_size: 16
                    input_activations:
                        num_bits: 4
                        type: float
                        strategy: tensor_group
                        dynamic: local
                        symmetric: true
                        group_size: 16
                    targets: ["Linear"]
    """

model = SparseAutoModelForCausalLM.from_pretrained(
    source_model_path,
    torch_dtype="auto"
)

tokenizer = AutoTokenizer.from_pretrained(source_model_path)

calib_dataset = load_dataset(
    "json",
    data_files=calib_dataset_path,
    split="train"
)

def convert_calib_dataset(sample):
    input_ids = sample["tok_input"]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids)
    }

calib_dataset = calib_dataset.map(
    convert_calib_dataset,
    remove_columns=calib_dataset.column_names
)

num_calibration_samples = min(num_calibration_samples, len(calib_dataset))

recipe = ""
if calibration_target == "fp8":
    recipe =  recipe_fp8
elif calibration_target == "fp4":
    recipe = recipe_fp4

oneshot(
    model=model,
    output_dir=target_model_path,
    dataset=calib_dataset,
    recipe=recipe,
    max_seq_length=max_sequence_length,
    num_calibration_samples=num_calibration_samples,
    save_compressed=True,
)
