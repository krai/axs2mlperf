# Quick Demo with Docker

Below is a self-contained demonstration of our onnxruntime workflow, which installs `axs` and downloads relevant dependencies. For object detection, it downloads the SSD-ResNet34 model, the RetinaNet model, the original COCO dataset and a short partial resized subset of 20 images. For image classification, it downloads the ResNet50 model, the original ImageNet dataset and a short partial resized subset of 20 images. For large language models, it downloads the Bert Large model and the original squad v1.1 dataset.

Download the [Dockerfile](Dockerfile).
```
wget -O Axs_Dockerfile https://raw.githubusercontent.com/krai/axs2mlperf/master/demo/Dockerfile
```

Build the Docker image. It takes ~30 minutes on our server and is ~9.51GB in size.
```
time docker build -t axs:benchmarks -f Axs_Dockerfile .
```

### SSD-ResNet34
Launch a short accuracy run of the SSD-ResNet34 model.
```
docker run -it --rm axs:benchmarks -c "time axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_dataset_size=20 , get accuracy"
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

### RetinaNet
Launch a short accuracy run of the RetinaNet model.
```
docker run -it --rm axs:benchmarks -c "time axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_dataset_size=20,model_name=retinanet_coco , get accuracy"
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

### ResNet50
Launch a short accuracy run of the ResNet50 model.
```
docker run -it --rm axs:benchmarks -c "time axs byquery loadgen_output,classified_imagenet,framework=onnx,loadgen_dataset_size=20  , get accuracy"
```
<details>
mAP value and run time
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
mAP value and run time
<pre>
85.0

real    0m30.967s
user    3m2.495s
sys     0m5.295s
</pre>
</details>

The mAP value and run time should be printed after a successful run. To install `axs` locally and to explore its full potential please check out the documentation of the [object detection](../object_detection_onnx_loadgen_py/README.md), [image classification](../image_classification_onnx_loadgen_py/README.md), and [large language model](../bert_squad_onnxruntime_loadgen_py/README.md) pipelines.

