#!/bin/bash

wget https://raw.github.com/lehmannro/assert.sh/v1.1/assert.sh
source assert.sh

_IMAGE_NAME=${IMAGE_NAME:-axs:benchmarks.test}

IS_SERVER=${IS_SERVER:-false}



run_docker () {
    if [[ ${IS_SERVER} = true ]]
    then 
	    docker exec git_bot $1
    else
            docker run -it --rm ${_IMAGE_NAME} -c "$1"
    fi
}

if [[ ${IS_SERVER} = true ]]
then
	docker run --name git_bot --entrypoint tail --privileged -t ${_IMAGE_NAME} -f /dev/null &
fi

cmd=$(run_docker "axs byquery loadgen_output,task=object_detection,framework=onnxrt,loadgen_dataset_size=20,model_name=retinanet_openimages , get accuracy")
echo "Accuracy(20 samples): ${cmd}"
num=${cmd:0:2}
assert "echo ${num}" "52"
assert_end object_detection_retinanet_openimages_benchmark


cmd=$(run_docker "axs byquery loadgen_output,task=object_detection,framework=onnxrt,loadgen_dataset_size=20,model_name=retinanet_coco , get accuracy")
echo "Accuracy(20 samples): ${cmd}"
num=${cmd:0:2}
assert "echo ${num}" "34"
assert_end object_detection_retinanet_coco_benchmark


cmd=$(run_docker "axs byquery loadgen_output,task=image_classification,framework=onnxrt,loadgen_dataset_size=20 , get accuracy")
echo "Accuracy(20 samples): ${cmd}"
num=${cmd:0:2}
assert "echo ${num}" "85"
assert_end image_classification_benchmark

cmd=$(run_docker "axs byquery loadgen_output,task=bert,framework=onnxrt,loadgen_dataset_size=20 , get accuracy")
echo "Accuracy(20 samples): ${cmd}"
num=${cmd:0:2}
assert "echo ${num}" "85"
assert_end bert_benchmark

rm assert.sh


if [[ ${IS_SERVER} = true ]]
then
	docker stop git_bot && docker rm git_bot
fi
