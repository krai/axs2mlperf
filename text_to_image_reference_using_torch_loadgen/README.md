
# MLPerf Inference - Text to Image - ifferent backend(pytorch, pytorch-dist, debug)
This implementation runs text to image model with the different backend.

Currently it supports the following models:
- stable-diffusion-xl

## Accuracy Full Run(5000 images):
- SingleStream
Accuracy Command:
```
axs byquery loadgen_output,task=text_to_image,framework=torch,profile=stable-diffusion-xl-pytorch,dtype=fp16,loadgen_scenario=SingleStream,accuracy+

```
Accuracy:
```
axs byquery loadgen_output,task=text_to_image,framework=torch,profile=stable-diffusion-xl-pytorch,dtype=fp16,loadgen_scenario=SingleStream,accuracy+ , get accuracy_dict
{'fid': 23.61146873078252, 'clip': 31.749096275568007}
````

```
axs byquery loadgen_output,task=text_to_image,framework=torch,profile=stable-diffusion-xl-pytorch,dtype=fp16,loadgen_scenario=SingleStream,accuracy+ , get fid
23.61146873078252
``

```
axs byquery loadgen_output,task=text_to_image,framework=torch,profile=stable-diffusion-xl-pytorch,dtype=fp16,loadgen_scenario=SingleStream,accuracy+ , get clip
31.749096275568007

```

## Perfomance Full Run
- SingleStream
Performance Command
```
axs byquery loadgen_output,task=text_to_image,framework=torch,profile=stable-diffusion-xl-pytorch,dtype=fp16,loadgen_scenario=SingleStream,loadgen_target_latency=6,loadgen_mode=PerformanceOnly,sut_name=7920t-kilt-onnxruntime_gpu
```
Performance
```
axs byquery loadgen_output,task=text_to_image,framework=torch,profile=stable-diffusion-xl-pytorch,dtype=fp16,loadgen_scenario=SingleStream,loadgen_target_latency=6,loadgen_mode=PerformanceOnly,sut_name=7920t-kilt-onnxruntime_gpu , get performance
VALID : _Early_stopping_90th_percentile_estimate=4621.172 (milliseconds)

```
