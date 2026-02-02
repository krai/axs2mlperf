"""
    A package for installing and running the mlperf_inf_mm_q3vl utility from MLCommons for reference benchmarking Qwen3 model.

    The main mode of operation assumes that a dockerized server is run first in a parallel session.
    Which will likely need a pre-downloaded Qwen3 model.

    [Session 1]$ docker run --gpus all -v /mnt/data:/mnt/data -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas -p 8000:8000 --ipc=host vllm/vllm-openai:latest --served-model-name Qwen/Qwen3-VL-235B-A22B-Instruct --model /mnt/data/krai/mlcommons-storage/models/vlm_model/Qwen3-VL-235B-A22B-Instruct --tensor-parallel-size 8 --limit-mm-per-prompt.video 0 --no-enable-prefix-caching

    Note: From experience, this can take anywhere between 12 and 75 minutes to load all the model's shards into memory.
    Note that if the model is preloaded, you need both --model (containing local path to the model) and --served-model-name (to give it the name).
    Otherwise --model Qwen/Qwen3-VL-235B-A22B-Instruct would attempt to download the whole 440G model into .cache/huggingface .
    Do not start the actual test before the server is ready to accept requests.

    In the second session it is advisable to run a few intermediate installs first:

        # Depending on the system, this may cause detection (quick), install-from-precompiled (medium) or install-from-sources (slowish) :
    [Session 2]$ axs byquery shell_tool,can_python,desired_python_version===3.12

        # Get the axs2mlperf repo that contains our reference benchmark
    [Session 2]$ axs byquery git_repo,collection,repo_name=axs2mlperf

        # Now pull all the deps and build the mlperf_inf_mm_q3vl utility itself:
    [Session 2]$ axs byquery shell_tool,can_benchmark_qwen3

        # If the server looks healthy, run the Accuracy benchmark:
    [Session 2]$ axs byquery loadgen_output,task=qwen3,framework=reference,loadgen_mode=AccuracyOnly

        # Assess the accuracy by processing the (cached) accuracy log (takes about 2 min, does not require a GPU) :
    [Session 2]$ ACCURACY_LOG_PATH=`axs byquery loadgen_output,task=qwen3,framework=reference,loadgen_mode=AccuracyOnly , get_path mlperf_log_accuracy.json`
    [Session 2]$ time axs byquery shell_tool,can_benchmark_qwen3 , run --cmd_key=accuracy_report --accuracy_log_path=$ACCURACY_LOG_PATH

        # Run the Performance benchmark:
    [Session 2]$ axs byquery loadgen_output,task=qwen3,framework=reference,loadgen_mode=PerformanceOnly

        # Assess the performance by processing the (cached) logs:
    [Session 2]$ axs byquery loadgen_output,task=qwen3,framework=reference,loadgen_mode=PerformanceOnly , get performance
"""
