# Quick Demo with Docker

Below is a self-contained demonstration of our onnxruntime workflow, which installs `axs` and downloads relevant dependencies. For object detection, it downloads the SSD-ResNet34 model, the RetinaNet model, the original COCO dataset and a short partial resized subset of 20 images. For image classification, it downloads the ResNet50 model, the first 500 images in the ImageNet dataset and a short partial resized subset of 20 images. For language models, it downloads the Bert Large model and the original squad v1.1 dataset.

Download the [Dockerfile](Dockerfile).
```
wget -O Axs_Dockerfile https://raw.githubusercontent.com/krai/axs2mlperf/master/demo/Dockerfile
```

Build the Docker image. It takes ~30 minutes on our server and is ~9.51GB in size.
```
time docker build -t axs:benchmarks -f Axs_Dockerfile .
```

## Measure Accuracy

### ResNet50
Launch a short accuracy run of the ResNet50 model.
```
docker run -it --rm axs:benchmarks -c "time axs byquery loadgen_output,classified_imagenet,framework=onnx,loadgen_dataset_size=20  , get accuracy"
```
<details>
accuracy and run time
<pre>
85.0

real    0m1.099s
user    0m5.070s
sys     0m2.685s
</pre>
</details>

### Bert Large
Launch a short accuracy run of the Bert Large model.
```
docker run -it --rm axs:benchmarks -c "time axs byquery loadgen_output,bert_squad,framework=onnx,loadgen_dataset_size=20  , get accuracy"
```
<details>
accuracy and run time
<pre>
85.0

real    0m30.967s
user    3m2.495s
sys     0m5.295s
</pre>
</details>

### RetinaNet
Launch a short accuracy run of the RetinaNet model.
```
docker run -it --rm axs:benchmarks -c "time axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_dataset_size=20,model_name=retinanet_coco , get mAP"
```
<details>
mAP value and run time
<pre>
34.671

real    0m20.131s
user    2m24.876s
sys     0m3.220s
</pre>
</details>

### SSD-ResNet34
Launch a short accuracy run of the SSD-ResNet34 model.
```
docker run -it --rm axs:benchmarks -c "time axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_dataset_size=20 , get mAP"
```
<details>
mAP value and run time
<pre>
22.852

real    0m26.530s
user    3m14.439s
sys     0m2.866s
</pre>
</details>

The mAP/accuracy and run time should be printed after a successful run. To install `axs` locally and to explore its full potential please check out the documentation of the [object detection](../object_detection_onnx_loadgen_py/README.md), [image classification](../image_classification_onnx_loadgen_py/README.md), and [language model](../bert_squad_onnxruntime_loadgen_py/README.md) pipelines.


## Measure Performance

Two important facts should be taken into account when benchmarking in performance mode:
1. There is no way to measure mAP (LoadGen's constraint)
2. You need to "guess" the Query-Per-Second (QPS) of your system. The idea is to set the `loadgen_target_qps` parameter as close to the actual performance of the machine as possible. Then, loadgen will issue enough queries such that the run will takes ~10 minutes.

Below details the steps of launching a performance run for the ResNet50 model, it will be similar for other models. (Note that for ResNet50, by default, we uses the first 500 images in the ImageNet dataset only.) We would recommend conducting the run with `axs` installed locally, but it is possible to do it within docker as well.

First, get an idea of the performance of the System-Under-Test (SUT).
```
docker run -it --rm axs:benchmarks -c "time axs byquery loadgen_output,classified_imagenet,framework=onnx,loadgen_dataset_size=20,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_target_qps=1 , get performance"
```

For an average CPU SUT, it will start printing the latency. `Ctr-C` after we get an idea about the latency of the system.
```
p[batch of 1] inference=6.90 ms
p[batch of 1] inference=6.99 ms
p[batch of 1] inference=6.95 ms
...
```

For a GPU system, see details.
<details>
For a GPU system, it will probably finish the run much quicker. Assuming the use of nvidia gpu, to check the GPU is visible within the docker container:
<pre>
docker run -it --rm --privileged --gpus all axs:benchmarks -c "nvidia-smi"
</pre>
Or,
<pre>
docker run -it --rm --privileged --gpus all axs:benchmarks -c "time axs byquery shell_tool,can_gpu , run"
</pre>

To run with GPU:
<pre>
docker run -it --rm --privileged --gpus all axs:benchmarks -c "time axs byquery loadgen_output,classified_imagenet,framework=onnx,loadgen_dataset_size=20,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_target_qps=1,verbosity=1,execution_device=gpu,cuda , get performance"
</pre>
</details>

In the above CPU example, the estimated QPS for the system would then be `1/7ms = 143 qps`. To conduct a proper run, we recommend removing the `--rm` flag such that we will not lost too many progress, and mount a volume with the `--volume` flag to export the resultant log. Remove `verbosity` as well.

Create a log folder in the current directory.
```
mkdir axs_logs && sudo chmod 777 axs_logs
```

Run.
```
docker run -it --name axs --volume ./axs_logs:/home/krai/logs axs:benchmarks -c "time axs byquery loadgen_output,classified_imagenet,framework=onnx,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=500,loadgen_buffer_size=1024,verbosity=1,loadgen_target_qps=143 , get performance"
```

Output.
```
Session execution provider:  ['CPUExecutionProvider']
Device: CPU
B500llllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll
...
Q94380pppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp
...
112.187
real    14m2.480s
user    336m34.121s
sys     0m5.368s
```

To view the log, commit the container to a new image.
```
docker commit axs axs:imageclassification
```

Enter the container.
```
docker run -it --rm -v ./axs_logs:/home/krai/logs --privileged --entrypoint /bin/bash axs:imageclassification
```

The MLPerf logs are stored in `$(axs byquery loadgen_output,classified_imagenet,framework=onnx,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline , get_path)`, to view the summary log:
```
krai@3342049c8b4d:~/work_collection$ vi $(axs byquery loadgen_output,classified_imagenet,framework=onnx,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline , get_path)/mlperf_log_summary.txt
...

================================================
MLPerf Results Summary
================================================
SUT name : PySUT
Scenario : Offline
Mode     : PerformanceOnly
Samples per second: 112.187
Result is : VALID
  Min duration satisfied : Yes
  Min queries satisfied : Yes
  Early stopping satisfied: Yes

================================================
Additional Stats
================================================
Min latency (ns)                : 47998898
Max latency (ns)                : 841276794720
...
```

Save the logs to the host machine.
```
cp -r $(axs byquery loadgen_output,classified_imagenet,framework=onnx,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline , get_path) /home/krai/logs
```
