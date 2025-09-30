from safetensors import safe_open
import torch
import math
from collections import defaultdict
from pathlib import Path

def get_dtypes_used(__entry__=None):
    files = list(Path(__entry__.get_path()).glob("*.safetensors"))

    param_counts = defaultdict(int)

    for fname in files:
        with safe_open(fname, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                dtype = str(tensor.dtype)
                shape = tensor.shape
                num_params = math.prod(shape)
                param_counts[dtype] += num_params

    return dict(param_counts)
