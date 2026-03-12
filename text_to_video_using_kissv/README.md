# KISS-V: Krai Inference Serving Solution for Video

## Prerequisites
* Docker must be installed.
* The user running benchmarks must be included in the `docker` group.

## Download the KISS-V (MLPerf edition) Docker image

### H200
```
docker pull krai4ai/kiss-v_mlperf:h200
```

## Install [Krai](https://krai.ai)'s [axs](https://github.com/krai/axs) automation technology

### Define a local workspace directory

Use `$HOME`:
```
export AXS_WORK_DIR=${HOME}
```

or e.g.:
```
export AXS_WORK_DIR=/mnt/data/krai/${USER}
```

### Clone

Clone the AXS repository under `${AXS_WORK_DIR}`:
```
git clone --branch master https://github.com/krai/axs ${AXS_WORK_DIR}/axs
```

### Init

Define environment variables in your `~/.bashrc` e.g.:
```
echo "

# AXS.
export AXS_WORK_DIR=/mnt/data/krai/${USER}
export AXS_WORK_COLLECTION=${AXS_WORK_DIR}/work_collection
export PATH=${AXS_WORK_DIR}/axs:${PATH}

" >> ~/.bashrc
```

### Test
```
source ~/.bashrc
axs version
```

### Import public AXS repositories

Import the required public repos into your work collection:
```
axs byquery git_repo,collection,repo_name=axs2mlperf
```


## Benchmark

See below example commands for benchmarking the Text-to-Video task under:
- the Closed/Open divisions;
- the SingleStream/Offline scenarios;
- the Accuracy/Performance/Compliance modes.

See commands used for actual submission runs under the corresponding
subdirectories of `<division>/<submitter>/results/`.

### Closed

#### SingleStream

##### Accuracy
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv,\
loadgen_scenario=SingleStream,loadgen_mode=AccuracyOnly,loadgen_target_latency=57000,\
nodes=1,node_idx=0,master_addr=localhost,hosts=localhost,dit_ring_para=4,sage_offset_layers=4,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200
```

##### Performance
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv,\
loadgen_scenario=SingleStream,loadgen_mode=PerformanceOnly,loadgen_target_latency=57000,\
nodes=1,node_idx=0,master_addr=localhost,hosts=localhost,dit_ring_para=4,sage_offset_layers=4,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200
```

##### Compliance

###### TEST04
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv,\
loadgen_scenario=SingleStream,loadgen_mode=PerformanceOnly,loadgen_target_latency=57000,\
loadgen_compliance_test=TEST04,\
nodes=1,node_idx=0,master_addr=localhost,hosts=localhost,dit_ring_para=4,sage_offset_layers=4,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200
```

#### Offline

##### Accuracy
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv,\
loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,loadgen_target_qps=0.02,\
nodes=1,node_idx=0,master_addr=localhost,hosts=localhost,dit_ring_para=4,sage_offset_layers=4,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200
```

##### Performance
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv,\
loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=0.02,\
nodes=1,node_idx=0,master_addr=localhost,hosts=localhost,dit_ring_para=4,sage_offset_layers=4,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200
```

##### Compliance

###### TEST04
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv,\
loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=0.02,\
loadgen_compliance_test=TEST04,\
nodes=1,node_idx=0,master_addr=localhost,hosts=localhost,dit_ring_para=4,sage_offset_layers=4,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200
```


### Open

#### SingleStream

##### Accuracy
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv_fast,\
loadgen_scenario=SingleStream,loadgen_mode=AccuracyOnly,loadgen_target_latency=44000,\
nodes=1,node_idx=0,master_addr=localhost,hosts=localhost,dit_ring_para=4,sage_offset_layers=4,\
division=open,caching_strategy=mag,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200
```

##### Performance
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv_fast,\
loadgen_scenario=SingleStream,loadgen_mode=PerformanceOnly,loadgen_target_latency=44000,\
nodes=1,node_idx=0,master_addr=localhost,hosts=localhost,dit_ring_para=4,sage_offset_layers=4,\
division=open,caching_strategy=mag,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200
```

#### Offline

##### Accuracy
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv_fast,\
loadgen_scenario=Offline,loadgen_mode=AccuracyOnly,loadgen_target_qps=0.02,\
nodes=1,node_idx=0,master_addr=localhost,hosts=localhost,dit_ring_para=4,sage_offset_layers=4,\
division=open,caching_strategy=mag,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200
```

##### Performance
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,sut_name=h200_n1-kissv_fast,\
loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=0.02,\
nodes=1,node_idx=0,master_addr=localhost,hosts=localhost,dit_ring_para=4,sage_offset_layers=4,\
division=open,caching_strategy=mag,\
docker_image_name=krai4ai/kiss-v_mlperf,docker_image_tag=h200
```
