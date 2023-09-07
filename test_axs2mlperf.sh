#!/bin/bash

source assert.sh

if [ "$ONNX_DETECTION_SSD_COCO" == "on" ] || [ "$ONNX_DETECTION_RETINANET_COCO" == "on" ] || [ "$ONNX_DETECTION_RETINANET_OPENIMAGES" == "on" ]; then
    if [ "$ONNX_DETECTION_SSD_COCO" == "on" ]; then
        axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu
        ACCURACY_OUTPUT=`axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu , get accuracy`
        echo "Accuracy: $ACCURACY_OUTPUT"
        assert 'echo `axs func round $ACCURACY_OUTPUT 0`' '23.0'
        axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu --- , remove
        assert_end object_detection_onnx_loadgen_py_ssd_resnet34

        axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,loadgen_target_qps=35,loadgen_min_duration_s=60,loadgen_max_duration_s=60,loadgen_count_override=51,execution_device=cpu , get performance
        axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,loadgen_target_qps=35,loadgen_min_duration_s=60,loadgen_max_duration_s=60,loadgen_count_override=51,execution_device=cpu --- , remove
    else
        echo "Skipping the ONNX_DETECTION_SSD_COCO test"
    fi
    if [ "$ONNX_DETECTION_RETINANET_OPENIMAGES" == "on" ]; then
        axs byquery extracted,openimages_annotations,v2_1
        axs byquery downloaded,openimages_mlperf
        axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=retinanet_openimages,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu
        ACCURACY_OUTPUT=`axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=retinanet_openimages,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu , get accuracy`
        echo "Accuracy: $ACCURACY_OUTPUT"
        assert 'echo `$ACCURACY_OUTPUT`' '52.98'
        axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=retinanet_openimages,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu --- , remove
        axs byquery downloaded,openimages_mlperf --- , remove
        axs byquery extracted,openimages_annotations,v2_1 --- , remove
        assert_end object_detection_onnx_loadgen_py_retinanet_openimages
    else
        echo "Skipping the ONNX_DETECTION_RETINANET_OPENIMAGES test"
    fi
fi

