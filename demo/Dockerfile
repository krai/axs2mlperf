FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]

ENTRYPOINT ["/bin/bash", "-c"]

# Download system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update -y\
 && apt install -y apt-utils\
 && apt upgrade -y\
 && apt install -y\
 git wget zip libz-dev\
 python3 python3-pip\
 vim sudo curl\
 && apt clean

# Create user 'krai' in group 'kraig'.
ARG GROUP_ID=1500
ARG USER_ID=2000
RUN groupadd -g ${GROUP_ID} kraig\
 && useradd -u ${USER_ID} -g kraig --create-home --shell /bin/bash krai
USER krai:kraig
RUN git config --global user.name "krai"\
 && git config --global user.email "info@krai.ai"

WORKDIR /home/krai

ARG BRANCH=stable

# Install the kernel of the X workflow automation framework.
RUN git clone --branch ${BRANCH} https://github.com/krai/axs
ENV PATH="$PATH:/home/krai/axs"

# Import this repository into your work_collection
RUN axs byquery git_repo,collection,repo_name=axs2mlperf,checkout=${BRANCH}

# Download model/dataset/runtime dependencies and perform two quick accuracy runs on SSD-ResNet34, RetinaNet, ResNet50, Bert Large.
# Object Detection
RUN axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_dataset_size=20 , get accuracy
RUN axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_dataset_size=20,model_name=retinanet_coco , get accuracy
# Image Classification
RUN axs byquery loadgen_output,classified_imagenet,framework=onnx,loadgen_dataset_size=20 , get accuracy
# Large Language Model
RUN axs byquery tokenized,squad_v1_1,calibration=no
RUN axs byquery loadgen_output,bert_squad,framework=onnx , get accuracy

# Remove the generated experiments to start at a clean state
RUN axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_dataset_size=20 , remove
RUN axs byquery loadgen_output,detected_coco,framework=onnx,loadgen_dataset_size=20,model_name=retinanet_coco , remove
RUN axs byquery loadgen_output,classified_imagenet,framework=onnx,loadgen_dataset_size=20 , remove
RUN axs byquery loadgen_output,bert_squad,framework=onnx , remove

# Provide a place for exporting files to host
RUN mkdir /home/krai/logs

CMD [ "/bin/bash" ]
