from pathlib import Path
import yaml

def generate_inference_config_yaml(
    output_dir,
    inference_config_filename,
    height,
    width,
    num_frames,
    fps,
    sample_steps,
    seed,
    guidance_scale,
    guidance_scale_2,
    boundary_ratio,
    negative_prompt
):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / inference_config_filename

    data = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "fps": fps,
        "sample_steps": sample_steps,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "guidance_scale_2": guidance_scale_2,
        "boundary_ratio": boundary_ratio,
        "negative_prompt": negative_prompt
    }

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True
        )

    return str(path.resolve())

def get_params(params_dict):
    params = []
    for key, value in params_dict.items():
        params.append(f"--{key} {value}")
    return " ".join(params)