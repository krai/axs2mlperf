# Quantizing a Llama model

## Define a local workspace directory
```
export WORKSPACE_DIR=/workspace
mkdir -p ${WORKSPACE_DIR}
mkdir -p ${WORKSPACE_DIR}/${USER}
```

## Install KRAI [AXS](https://github.com/krai/axs)
### Clone
Clone the AXS repository under `${WORKSPACE_DIR}/$USER`:
```
git clone https://github.com/krai/axs ${WORKSPACE_DIR}/$USER/axs
```
### Init
Define environment variables in your `~/.bashrc`:
```
echo "

# AXS.
export WORKSPACE_DIR=/workspace
export PATH=${WORKSPACE_DIR}/${USER}/axs:${PATH}
export AXS_WORK_COLLECTION=${WORKSPACE_DIR}/${USER}/work_collection

" >> ~/.bashrc
```
### Test
```
source ~/.bashrc
axs version
```
## Setup dependencies
### Install KRAI's `axsmlperf` repo
```
axs byquery git_repo,collection,repo_name=axs2mlperf
```
### Install mlc-r2-downloader
```
axs byquery downloaded,mlc_r2_downloader
```
### Setup the HuggingFace token
```
export HF_TOKEN=...
```
### Download the original Llama model
```
export MODEL_NAME=llama3_1
export MODEL_VARIANT=8b
axs byquery downloaded,hf_model,model_family=${MODEL_NAME},variant=${MODEL_VARIANT},hf_token=${HF_TOKEN}
```

### [Optional] Install Python 3.11
The commands below require a Python version < 3.12, so if you do not have such a version installed on your system, run
```
axs byquery installed,python3,minor_version=11
axs byquery shell_tool,can_python,desired_python_version===3.11
```

## Convert the original bf16 model to fp16
```
axs byquery converted,method=transformers,model_name=${MODEL_NAME},model_variant=${MODEL_VARIANT}
```
[Optional] You can use a pre-downloaded model by running
```
axs byquery converted,method=transformers,source_model_path=...
```

## Quantize the model
```
axs byquery quantized,method=llm_compressor,model_name=${MODEL_NAME},model_variant=${MODEL_VARIANT}
```
[Optional] You can use a pre-converted to `fp16` or the original `bf16` model by running
```
axs byquery quantized,method=llm_compressor,source_model_path=...
```

# Operations breakdown
## Original Llama3.1-8b model
```
Total parameters by dtype:
  torch.bfloat16 : 8,030,261,696
```
## RedHat bf16/fp8 model
```
Total parameters by dtype:
  torch.bfloat16 : 1,050,939,840
  torch.float8_e4m3fn : 6,979,321,856
```
## Krai-converted tp fp16 and quantized to fp8
```
Total parameters by dtype:
  torch.float16 : 1,050,939,840
  torch.float8_e4m3fn : 6,979,321,856
```