from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys

source_model_path = sys.argv[1]
target_model_path = sys.argv[2]

model = AutoModelForCausalLM.from_pretrained(
    source_model_path,
    torch_dtype="auto",
    device_map="cpu"
)

print("Casting to fp16...")
model = model.to(torch.float16)

print(f"Saving FP16 model to {target_model_path} ...")
model.save_pretrained(
    target_model_path,
    safe_serialization=True
)

tokenizer = AutoTokenizer.from_pretrained(source_model_path)
tokenizer.save_pretrained(target_model_path)