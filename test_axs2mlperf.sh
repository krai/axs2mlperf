#!/bin/bash

source assert.sh

if [ "$ONNX_DETECTION" == "on" ]; then
    axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu
    ACCURACY_OUTPUT=`axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu , get accuracy`
    echo "Accuracy: $ACCURACY_OUTPUT"
    assert 'echo $ACCURACY_OUTPUT' '23.015'
    axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu --- , remove
    assert_end object_detection_onnx_loadgen_py
fi
