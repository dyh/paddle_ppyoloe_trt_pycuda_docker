ARG CUDA_VERSION=11.6.2
ARG OS_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}

# 由sh切换到bash
SHELL ["/bin/bash", "-c"]

# 修改时区
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# apt换阿里源
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
	&& apt clean \
	&& apt update \
	&& apt install

# 安装编译工具
RUN apt install -y --no-install-recommends \
unzip \
pkg-config \
wget \
git \
sudo \
ssh \
libssl-dev \
build-essential \
cmake

# 安装python3.8、pip、venv
RUN apt install -y --no-install-recommends \
      python3.8 \
      python3-pip \
      python3.8-dev \
      python3-wheel &&\
      cd /usr/local/bin && \
      ln -s /usr/bin/python3.8 python &&\
      ln -s /usr/bin/pip3 pip


# 安装 tensorrt
ARG TRT_VERSION=8.4.1.5

RUN v="${TRT_VERSION%.*}-1+cuda${CUDA_VERSION%.*}" && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update && \
    sudo apt-get install -y libnvinfer8=${v} libnvonnxparsers8=${v} libnvparsers8=${v} libnvinfer-plugin8=${v} \
        libnvinfer-dev=${v} libnvonnxparsers-dev=${v} libnvparsers-dev=${v} libnvinfer-plugin-dev=${v} \
        python3-libnvinfer=${v} tensorrt


# 升级 pip 包们
RUN python -m pip install --upgrade pip setuptools

RUN pip install paddle2onnx onnx pycuda nvidia-pyindex

# 先安装 nvidia-pyindex
RUN pip install onnx-graphsurgeon onnxsim


# 清理安装文件
RUN apt clean -y \
	&& apt autoclean -y \
	&& apt autoremove -y \
	&& apt update