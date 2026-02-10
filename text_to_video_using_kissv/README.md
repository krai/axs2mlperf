# MLPerf Inference - text_to_video task

## Prerequisites
* Docker must be installed.
* The user running benchmarks must be incuded in the `docker` group.

## Define a local workspace directory
```
export WORKSPACE_DIR=/workspace
mkdir -p ${WORKSPACE_DIR}
mkdir -p ${WORKSPACE_DIR}/${USER}
```

## Install KRAI [AXS](https://github.com/krai/axs)

### Clone

Clone the AXS repository under `${WORKSPACE_DIR}/$USER`:
```
git clone https://github.com/krai/axs ${WORKSPACE_DIR}/$USER/axs
```

### Init

Define environment variables in your `~/.bashrc`:
```
echo "

# AXS.
export WORKSPACE_DIR=/workspace
export PATH=${WORKSPACE_DIR}/${USER}/axs:${PATH}
export AXS_WORK_COLLECTION=${WORKSPACE_DIR}/${USER}/work_collection

" >> ~/.bashrc
```

### Test
```
source ~/.bashrc
axs version
```

## Import public AXS repositories

Import the required public repos into your work collection:

```
axs byquery git_repo,collection,repo_name=axs2mlperf
```

## Download artifacts

### KISS-V docker image
Download the image
```
docker pull kiss-v:latest
```

## Benchmark

See below example commands for the Server/Offline scenarios and under the Accuracy/Performance/Compliance modes.

See commands used for actual submission runs under the corresponding subdirectories of `results/`.

### Offline

#### Accuracy
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly
```

#### Performance
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=<desired qps>
```

### SingleStream

#### Accuracy
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,loadgen_scenario=SingleStream,loadgen_mode=AccuracyOnly
```

#### Performance
```
axs byquery loadgen_output,task=text_to_video,framework=kiss_v,loadgen_scenario=SingleStream,loadgen_mode=PerformanceOnly,loadgen_target_latency=<desired latency>
```