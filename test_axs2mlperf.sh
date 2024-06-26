#!/bin/bash

source assert.sh

if [ "$ONNX_DETECTION_SSD_COCO" == "on" ] || [ "$ONNX_DETECTION_RETINANET_COCO" == "on" ] || [ "$ONNX_DETECTION_RETINANET_OPENIMAGES" == "on" ]; then
    if [ "$ONNX_DETECTION_SSD_COCO" == "on" ]; then
        axs byquery loadgen_output,task=object_detection,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu
        export ACCURACY_OUTPUT=`axs byquery loadgen_output,task=object_detection,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu , get accuracy`
        echo "Accuracy: $ACCURACY_OUTPUT"
        assert 'echo `axs func round $ACCURACY_OUTPUT 0`' '23.0'
        axs byquery loadgen_output,task=object_detection,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu --- , remove

        axs byquery loadgen_output,task=object_detection,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,loadgen_target_qps=5,loadgen_min_duration_s=60,loadgen_max_duration_s=60,loadgen_query_count=51,execution_device=cpu , get performance
        axs byquery loadgen_output,task=object_detection,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,model_name=ssd_resnet34,loadgen_dataset_size=20,loadgen_buffer_size=100,loadgen_target_qps=5,loadgen_min_duration_s=60,loadgen_max_duration_s=60,loadgen_query_count=51,execution_device=cpu --- , remove

        axs byquery downloaded,onnx_model,model_name=ssd_resnet34 --- , remove
        axs byquery preprocessed,dataset_name=coco,resolution=1200,first_n=20 --- , remove
        axs byquery downloaded,coco_images --- , remove
        axs byquery extracted,coco_images --- , remove
        axs byquery downloaded,coco_annotation --- , remove
        axs byquery extracted,coco_annotation --- , remove

        assert_end object_detection_using_onnxrt_loadgen_ssd_resnet34
    else
        echo "Skipping the ONNX_DETECTION_SSD_COCO test"
    fi
    if [ "$ONNX_DETECTION_RETINANET_OPENIMAGES" == "on" ]; then
        #axs byquery extracted,openimages_annotations,v2_1
        #axs byquery downloaded,openimages_mlperf
        axs byquery loadgen_output,task=object_detection,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=retinanet_openimages,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu
        export ACCURACY_OUTPUT=`axs byquery loadgen_output,task=object_detection,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=retinanet_openimages,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu , get accuracy`
        echo "Accuracy: $ACCURACY_OUTPUT"
        assert "echo $ACCURACY_OUTPUT" '52.98'
        #assert "echo $ACCURACY_OUTPUT" '52.939' # for python3.6
        axs byquery loadgen_output,task=object_detection,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=retinanet_openimages,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu --- , remove
        axs byquery downloaded,openimages_mlperf --- , remove
        axs byquery extracted,openimages_annotations,v2_1 --- , remove
        axs byquery downloaded,openimages_annotations,v2_1 --- , remove
        axs byquery downloaded,onnx_model,model_name=retinanet_openimages --- , remove
        axs byquery preprocessed,dataset_name=openimages,resolution=800,first_n=20 --- , remove

        assert_end object_detection_using_onnxrt_loadgen_retinanet_openimages
    else
        echo "Skipping the ONNX_DETECTION_RETINANET_OPENIMAGES test"
    fi
fi

if [ "$ONNX_CLASSIFY" == "on" ] || [ "$TORCH_CLASSIFY" == "on" ] ; then
    if [ "$ONNX_CLASSIFY" == "on" ]; then
        axs byquery loadgen_output,task=image_classification,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=resnet50,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu
        export ACCURACY_OUTPUT=`axs byquery loadgen_output,task=image_classification,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=resnet50,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu , get accuracy `
        echo "Accuracy: $ACCURACY_OUTPUT"
        assert "echo $ACCURACY_OUTPUT" '85.0'

        axs byquery loadgen_output,task=image_classification,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=resnet50,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu --- , remove
        axs byquery downloaded,onnx_model,model_name=resnet50 --- , remove
        axs byquery preprocessed,dataset_name=imagenet,resolution=224,first_n=20 --- , remove

        assert_end image_classification_using_onnxrt_loadgen
    else
        echo "Skipping the ONNX_CLASSIFY test"
    fi
    if [ "$TORCH_CLASSIFY" == "on" ]; then
        axs byquery loadgen_output,task=image_classification,framework=pytorch,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=resnet50,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu
        export ACCURACY_OUTPUT=`axs byquery loadgen_output,task=image_classification,framework=pytorch,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=resnet50,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu , get accuracy`
        echo "Accuracy: $ACCURACY_OUTPUT"
        assert "echo $ACCURACY_OUTPUT" '75.0'

        axs byquery loadgen_output,task=image_classification,framework=pytorch,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,model_name=resnet50,loadgen_dataset_size=20,loadgen_buffer_size=100,execution_device=cpu --- , remove
        axs byquery preprocessed,dataset_name=imagenet,resolution=224,first_n=20 --- , remove

        assert_end image_classification_using_torch_loadgen
   else
       echo "Skipping the TORCH_CLASSIFY test"
   fi
fi

if [ "$ONNX_BERT_SQUAD" == "on" ]; then
    #axs byquery preprocessed,dataset_name=squad_v1_1
    axs byquery loadgen_output,task=bert,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,execution_device=cpu
    export ACCURACY_OUTPUT=`axs byquery loadgen_output,task=bert,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,execution_device=cpu , get accuracy_dict`
    echo "Accuracy: $ACCURACY_OUTPUT"
    assert 'echo $ACCURACY_OUTPUT' "{'exact_match': 85.0, 'f1': 85.0}"
    axs byquery loadgen_output,task=bert,framework=onnxrt,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,execution_device=cpu --- , remove
    axs byquery inference_ready,onnx_model,model_name=bert_large --- , remove
    axs byquery preprocessed,dataset_name=squad_v1_1 --- , remove
    assert_end bert_using_onnxrt_loadgen
else
    echo "Skipping the ONNX_BERT_SQUAD test"
fi
