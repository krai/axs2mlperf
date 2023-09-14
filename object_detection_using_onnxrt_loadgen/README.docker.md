# Quick Demo with Docker

Below is a self-contained demonstration of our object detection workflow, which installs `axs`, downloads some necessary Python packages, the SSD-ResNet34 model, the RetinaNet model, the original COCO dataset and a short partial resized subset of 20 images.

Download the [Dockerfile](https://github.com/krai/axs2mlperf/blob/master/object_detection_onnx_loadgen_py/Dockerfile).
```
wget -O Axs_Docker https://raw.githubusercontent.com/krai/axs2mlperf/master/object_detection_onnx_loadgen_py/Dockerfile
```

Build the Docker image. It takes ~12 minutes on our server and is ~4.89GB in size.
```
time docker build -t axs:object-detection -f Axs_Docker .
```

Launch a short accuracy run of the SSD-ResNet34 model.
```
docker run -it --rm axs:object-detection -c "time axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_dataset_size=20 , get accuracy"
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

Launch a short accuracy run of the RetinaNet model.
```
docker run -it --rm axs:object-detection -c "time axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_dataset_size=20,model_name=retinanet_coco , get accuracy"
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

The mAP value and run time should be printed after a successful run. To install `axs` locally and to explore its full potential please read [README.md](README.md).

