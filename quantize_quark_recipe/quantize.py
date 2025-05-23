import sys
import json
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

from quark.torch import ModelQuantizer, ModelExporter
from quark.torch.quantization import (Config, QuantizationConfig,
                                     FP8E4M3PerTensorSpec,
                                     load_quant_algo_config_from_file)
from quark.torch.export import ExporterConfig, JsonExporterConfig, OnnxExporterConfig

# Load the input parameters
input_parameters_file_path = sys.argv[1]
input_parameters = {}
with open(input_parameters_file_path) as f:
    input_parameters = json.load(f)

model_path                          = input_parameters["model_path"]
max_seq_len                         = int(input_parameters["max_seq_len"])
batch_size                          = int(input_parameters["batch_size"])
num_calib_data                      = int(input_parameters["num_calib_data"])
device                              = input_parameters["device"]
data_type                           = input_parameters["data_type"]
model_attn_implementation           = input_parameters["model_attn_implementation"]
dataset_name                        = input_parameters["dataset_name"]
dataset_path                        = input_parameters["dataset_path"]
output_dir                          = input_parameters["output_dir"]
quark_source_path                   = input_parameters["quark_source_path"]
quant_scheme                        = input_parameters["quant_scheme"]
group_size                          = int(input_parameters["group_size"])
kv_cache_dtype                      = input_parameters["kv_cache_dtype"]
fp8_attention_quant                 = bool(input_parameters["fp8_attention_quant"])
exclude_layers                      = input_parameters["exclude_layers"]
pre_quantization_optimization       = input_parameters["pre_quantization_optimization"]
pre_optimization_config_file_path   = input_parameters["pre_optimization_config_file_path"]
quant_algo                          = input_parameters["quant_algo"]
quant_algo_config_file_path         = input_parameters["quant_algo_config_file_path"]
model_type                          = input_parameters["model_type"]
group_size_per_layer                = input_parameters["group_size_per_layer"]
weight_matrix_merge                 = bool(input_parameters["weight_matrix_merge"])
export_weight_format                = input_parameters["export_weight_format"]
pack_method                         = input_parameters["pack_method"]
min_kv_scale                        = float(input_parameters["min_kv_scale"])
custom_mode                         = input_parameters["custom_mode"]

# Additional imports
sys.path.append(os.path.join(quark_source_path), 'examples/torch/language_modeling')
from llm_ptq.configuration_preparation import get_config, get_export_config
from llm_utils.model_preparation import MODEL_NAME_KV_LAYERS_MAP
from llm_utils.export_import_hf_model import export_hf_model

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device,
                                             torch_dtype=data_type, trust_remote_code=True,
                                             attn_implementation=model_attn_implementation)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=max_seq_len,
                                          padding_side="left", trust_remote_code=True,
                                          use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# Load the dataset and get calibration data.
samples = []

# OpenORCA dataset
if dataset_name == "openorca":
    from rouge_evaluate import prepare_openorca
    from torch.nn.functional import pad
    source_ids, source_lengths, target_ids, target_text = prepare_openorca(dataset_path)
    for idx in range(num_calib_data):
        input_length = source_lengths[idx]
        input_ids = torch.tensor(source_ids[idx], device=device, dtype=torch.int32).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int32, device=device)
        input_ids = pad(input_ids, (max_seq_len - input_length, 0, 0, 0), value=tokenizer.pad_token_id)
        attention_mask = pad(attention_mask, (max_seq_len - input_length, 0, 0, 0), value=0)
        samples.append({'input_ids': input_ids, 'attention_mask': attention_mask})

calib_dataloader = DataLoader(samples, batch_size=None, shuffle=False)
#calib_dataloader = DataLoader(samples, batch_size=batch_size, drop_last=True)

# Define global quantization config
quant_config = get_config(
    quant_scheme,
    group_size,
    model_path,
    kv_cache_dtype,
    fp8_attention_quant,
    exclude_layers if exclude_layers else None,
    pre_quantization_optimization if pre_quantization_optimization else None,
    pre_optimization_config_file_path if pre_quantization_optimization else None,
    quant_algo if quant_algo else None,
    quant_algo_config_file_path if quant_algo else None,
    model_type,
    group_size_per_layer if group_size else None
)

# Apply quantization.
quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model, calib_dataloader)

# Freeze quantized model to export.
freezed_model = quantizer.freeze(model)

# Define export config.
export_config = None
if weight_matrix_merge:
    json_export_config = JsonExporterConfig(
        weight_merge_groups=[["*up_proj", "*gate_proj"], ["*q_proj", "*k_proj", "*v_proj"]],
        weight_format=export_weight_format, pack_method=pack_method)
    export_config = ExporterConfig(
        json_export_config=json_export_config,
        onnx_export_config=OnnxExporterConfig())
else:
    json_export_config = JsonExporterConfig(weight_format=export_weight_format, pack_method=pack_method)
    export_config = ExporterConfig(
        json_export_config=json_export_config,
        onnx_export_config=OnnxExporterConfig())

if kv_cache_dtype:
    export_config.json_export_config.kv_cache_group = MODEL_NAME_KV_LAYERS_MAP[model_type]

export_config.json_export_config.min_kv_scale = min_kv_scale

exporter = ModelExporter(config=export_config, export_dir=output_dir)

with torch.no_grad():
    quant_config = get_config(
        quant_scheme,
        group_size,
        model_path,
        kv_cache_dtype,
        fp8_attention_quant,
        quantizer.config.exclude,
        pre_quantization_optimization,
        pre_optimization_config_file_path,
        quant_algo,
        quant_algo_config_file_path,
        model_type,
        group_size_per_layer,
    )
    export_hf_model(
        model,
        export_config,
        model_path,
        output_dir,
        quant_config,
        custom_mode=custom_mode)