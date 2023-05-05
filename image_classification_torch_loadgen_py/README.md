# MLPerf Inference - Image Classification - PYTORCH

This Python implementation runs PYTORCH models for Image Classification.

Currently it supports the following models:
- resnet50 on ImageNet dataset

## Prerequisites

This workflow is designed to showcase the `axs` workflow management system.
So the only prerequisite from the user's point of view is a sufficiently fresh version of `axs` system.

First, clone the `axs` repository.
```
git clone --branch stable https://github.com/krai/axs
```

Then, add the path to `bashrc`.
```
echo "export PATH='$PATH:$HOME/axs'" >> ~/.bashrc && \
source ~/.bashrc
```

Finally, import this repository into your `work_collection`
```
axs byquery git_repo,collection,repo_name=axs2mlperf,checkout=stable
```

The dependencies of various components (on Python code and external utilities) as well as interdependencies of the workflow's main components (original dataset, preprocessed dataset, model and its parameters) have been described in `axs`'s internal language to achieve the fullest automation we could.

Please note that due to this automation (automatic recursive installation of all dependent components) the external timing of the initial runs (when new components have to be downloaded and/or installed) may not be very useful. The internal timing as measured by the LoadGen API should be trusted instead, which is not affected by these changes in external infrastructure.


## Initial clean-up (optional)

In some cases it may be desirable to "start from a clean slate" - i.e. clean up all the cached `axs` entries,
which includes the model with weights, the original COCO dataset and its resized versions
(different models need different resizing resolutions), as well as all the necessary Python packages.

On the other hand, since all those components may take considerable time to be installed, we do not recommend cleaning up between individual runs.
The entry cache is there for a reason.

The following command effectively wipes off hours of downloading, compilation and/or installation:
```
axs work_collection , remove && \
axs byquery git_repo,collection,repo_name=axs2mlperf
```

## Performing a short Accuracy run (some parameters by default)

The following test run should trigger downloading and installation of the necessary Python packages, the default model (resnet50), the ImageNet dataset and the default dataset size(loadgen_dataset_size=20):
```
axs byquery loadgen_output,classified_imagenet,framework=pytorch,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline , get accuracy
```


## Performing a short Accuracy run (specifying the number of samples to run on)

The following test run should trigger downloading and installation of the necessary Python packages, the default model (resnet50), the ImageNet dataset and a short partial resized subset of 20 images:
```
axs byquery loadgen_output,classified_imagenet,framework=pytorch,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=20 , get accuracy
```
The accuracy value should be printed after a successful run.


## Performing a short Accuracy run (specifying the model)

The following test run should trigger (in addition to the above) downloading and installation of the resnet50 model:
```
axs byquery loadgen_output,classified_imagenet,framework=pytorch,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=20,model_name=resnet50 , get accuracy
```
The accuracy value should be printed after a successful run.


## Benchmarking resnet50 model in the Accuracy mode

The following command will run on the whole dataset of 50000 images used by the resnet50 model. Please note that depending on whether both the hardware and the software supports running on the GPU, the run may be performed either on the GPU or on the CPU. For running on the CPU it is necessary to add execution_device=cpu to the command.
(There are ways to constrain this to the CPU only.)

Example for 500 images:
```
axs byquery loadgen_output,classified_imagenet,framework=pytorch,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=500,loadgen_buffer_size=1024 , get accuracy_report
```
The accuracy value should be printed after a successful run.
<details><pre>
...
accuracy=75.200%, good=376, total=500
</pre></details>


```
axs byquery loadgen_output,classified_imagenet,framework=pytorch,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=500,loadgen_buffer_size=1024 , get accuracy
```
The accuracy value should be printed after a successful run.
<details><pre>
...
75.2
</pre></details>

## Benchmarking resnet50 model in the Performance mode

###Offline

You need to "guess" the `loadgen_target_qps` parameter, from which the testing regime will be generated in order to measure the actual QPS.

So `TargetQPS` is the input, whereas `QPS` is the output of this benchmark:
```
axs byquery loadgen_output,classified_imagenet,framework=pytorch,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=500,loadgen_buffer_size=1024,loadgen_target_qps=169 , get performance
```
Measured QPS:
```
...
171.966
```

###SingleStream

You need to set the `loadgen_target_latency` parameter.
```
axs byquery loadgen_output,classified_imagenet,framework=pytorch,loadgen_mode=PerformanceOnly,loadgen_scenario=SingleStream,loadgen_dataset_size=500,loadgen_buffer_size=1024,loadgen_target_latency=6 , get performance
```

```
...
6123339
```
