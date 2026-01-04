FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip3 install --upgrade pip

RUN pip3 install \
    runpod \
    torch \
    torchvision \
    diffusers \
    transformers \
    accelerate \
    pillow \
    safetensors \
    opencv-python

# Copia o código
COPY . /app
