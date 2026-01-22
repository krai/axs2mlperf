import json
import os

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

    for item in accuracy_log:
        prompt_idx = item["qsl_idx"]
        data_dump = item["data"]

        save_video(video_output_path, prompts[prompt_idx], data_dump)
    
    return video_output_path