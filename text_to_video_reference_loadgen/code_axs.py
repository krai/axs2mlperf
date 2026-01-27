from pathlib import Path
import yaml

def get_volumes(volumes_mapping, output_entry=None):
    volume_strings = []
    for i in range(0, len(volumes_mapping), 2):
        host_path = volumes_mapping[i]
        container_path = volumes_mapping[i+1]
        volume_strings.append(f"-v {host_path}:{container_path}")
    return " ".join(volume_strings)

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