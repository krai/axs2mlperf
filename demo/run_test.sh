#!/bin/bash

wget https://raw.github.com/lehmannro/assert.sh/v1.1/assert.sh
source assert.sh

_IMAGE_NAME=${IMAGE_NAME:-axs:benchmarks.test}

run_docker () {
    docker run -it --rm ${_IMAGE_NAME} -c "$1"
}


cmd=$(run_docker "time axs byquery loadgen_output,classified_imagenet,framework=onnxrt,loadgen_dataset_size=20  , get accuracy")
assert "echo ${cmd}" "85.0"

cmd=$(run_docker "time axs byquery loadgen_output,bert_squad,framework=onnxrt,loadgen_dataset_size=20  , get accuracy")
assert "echo ${cmd}" "85.0"

cmd=$(run_docker "time axs byquery loadgen_output,task=object_detection,framework=onnxrt,loadgen_dataset_size=20,model_name=retinanet_coco , get mAP")
assert "echo ${cmd}" "34.671"

cmd=$(run_docker "time axs byquery loadgen_output,task=object_detection,framework=onnxrt,loadgen_dataset_size=20 , get mAP")
assert "echo ${cmd}" "22.852"

assert_end benchmarks

