FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    sudo \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY wan-2.2-t2v-a14b/docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Create user with configurable UID/GID to match host user
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USER_NAME=user

RUN groupadd -g ${GROUP_ID} ${USER_NAME} || true && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash ${USER_NAME} && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to non-root user
USER ${USER_NAME}

# Set cache directory environment variables
ENV HF_HOME=/workspace/wan-2.2-t2v-a14b/.cache/huggingface
ENV HF_HUB_CACHE=/workspace/wan-2.2-t2v-a14b/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/wan-2.2-t2v-a14b/.cache/huggingface
ENV TORCH_HOME=/workspace/wan-2.2-t2v-a14b/.cache/torch
ENV TORCH_EXTENSIONS_DIR=/workspace/wan-2.2-t2v-a14b/.cache/torch_extensions

# Add VBench to Python path
ENV PYTHONPATH=/workspace/wan-2.2-t2v-a14b/submodules/VBench:${PYTHONPATH}

# Set working directory
WORKDIR /workspace/wan-2.2-t2v-a14b

# Default command
CMD ["/bin/bash"]