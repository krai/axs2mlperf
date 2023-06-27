Axs Bert Pruner
==============================
# Quick Demo with Docker

Below is a self-contained demonstration of our Axs Bert Pruner, which installs `axs`, [`bert pruner`](https://pypi.org/project/bert-pruners/), and downloads relevant dependencies in a docker container.

Download the [Dockerfile](Dockerfile).
```
wget -O Axs_Dockerfile https://raw.githubusercontent.com/krai/axs2mlperf/master/bert_pruner/Dockerfile
```

Build the Docker image.
```
time docker build --no-cache -t axs:bert_pruner -f Axs_Dockerfile .
```

Create a log folder in the current directory.
```
mkdir axs_logs && sudo chmod 777 axs_logs
```

Development Stage (Optional)
Only for development stage. Check out a different branch.
```
docker run -it --name axs_bert_pruner --volume ./axs_logs:/home/krai/logs axs:bert_pruner -c "cd $(axs byname axs2mlperf , get_path) && git checkout bert_pruner"
```
Or if you have axs installed and some local changes in axs2mlperf.
```
docker run -it --name axs_bert_pruner --volume ./axs_logs:/home/krai/logs --volume $(axs byname axs2mlperf , get_path):/home/krai/work_collection/axs2mlperf axs:bert_pruner
```

Run.
```
docker run -it --name axs_bert_pruner --volume ./axs_logs:/home/krai/logs axs:bert_pruner -c "axs byquery loadgen_output,bert_squad,framework=onnx,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=100,model_path=$(axs byquery bert_pruner_output,bert_squad,sparsity=0.5 , get_path) , get accuracy"
```

Save model (Optional)
```
docker run -it --name axs_bert_pruner --volume ./axs_logs:/home/krai/logs axs:bert_pruner -c "cp $(axs byquery bert_pruner_output,bert_squad,sparsity=0.5 , get_path) /home/krai/logs"
```

Clean up (Optional)
```
docker container rm axs_bert_pruner &&\
docker image rm axs:bert_pruner
```
