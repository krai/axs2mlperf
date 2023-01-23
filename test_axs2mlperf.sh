#!/bin/bash

source assert.sh

if [ "$ONNX_DETECTION" == "on" ]; then
    axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu
    ACCURACY_OUTPUT=`axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu , get accuracy`
    echo "Accuracy: $ACCURACY_OUTPUT"
    assert 'echo `axs func round $ACCURACY_OUTPUT 1`' '23.0'
    axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu --- , remove
    assert_end object_detection_onnx_loadgen_py

    axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,loadgen_target_qps=35,loadgen_min_duration_s=60,loadgen_max_duration_s=60,loadgen_count_override=51,execution_device=cpu , get performance
    axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,loadgen_target_qps=35,loadgen_min_duration_s=60,loadgen_max_duration_s=60,loadgen_count_override=51,execution_device=cpu --- , remove
fi

