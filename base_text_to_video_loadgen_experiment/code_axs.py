import json
import os
import re
from pathlib import Path

def save_video(video_output_path: str, prompt: str, data: str):
    if not os.path.exists(video_output_path):
        os.makedirs(video_output_path)

    file_path = os.path.join(video_output_path, f"{prompt}-0.mp4")
    with open(file_path, "wb") as f:
        f.write(bytes.fromhex(data))

def get_videos_path(output_path, videos_relative_path, dataset_path, accuracy_log_path):
    video_output_path = os.path.join(output_path, videos_relative_path)

    with open(accuracy_log_path) as f:
        accuracy_log = json.load(f)

    with open(dataset_path) as f:
        prompts = [line.strip() for line in f.readlines()]

    print(f"Saving videos to: {video_output_path}")
    for i, item in enumerate(accuracy_log):
        prompt_idx = item["qsl_idx"]
        data_dump = item["data"]

        save_video(video_output_path, prompts[prompt_idx], data_dump)
        print(f"Saved video {i} of {len(accuracy_log)}")
    print("All videos saved.")

    return video_output_path

def extract_overall_average(output_path):
    overall_average = -1.0

    ending = "_eval_results.json"
    output_folder = Path(output_path)
    report = None
    for file in output_folder.iterdir():
        if file.is_file() and file.name.endswith(ending):
            report = file
            break
    if report is None:
        return overall_average

    with open(report, 'r') as f:
        results = json.load(f)

    if results:
        total_score = 0
        num_dimensions = 0

        for _, value in sorted(results.items()):
            if isinstance(value, list) and len(value) > 0:
                score = value[0]
                if isinstance(score, (int, float)):
                    total_score += score
                    num_dimensions += 1

        if num_dimensions > 0:
            overall_average = total_score / num_dimensions

    return overall_average