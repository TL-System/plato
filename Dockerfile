# Download base image ubuntu 20.04
FROM ubuntu:20.04
LABEL maintainer="Baochun Li"
WORKDIR /root

ADD .bashrc /root/
COPY . /root/plato

RUN apt-get update \
    && apt-get install -y wget \
    && apt-get install -y vim \
    && apt-get install -y net-tools \
    && apt-get install -y git \
    && mkdir -p ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm -rf ~/miniconda3/miniconda.sh \
    && ~/miniconda3/bin/conda update -n base -c defaults conda \
    && ~/miniconda3/bin/conda init bash \
    && ~/miniconda3/bin/conda create -n pytorch python=3.8 \
    && ~/miniconda3/bin/conda install pytorch torchvision cpuonly -c pytorch -n pytorch \
    && ~/miniconda3/envs/pytorch/bin/pip install -r ~/plato/requirements.txt \
    && ~/miniconda3/bin/conda create -n mindspore python=3.7 \
    && ~/miniconda3/envs/mindspore/bin/pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.2.0-rc1/MindSpore/cpu/ubuntu_x86/mindspore-1.2.0rc1-cp37-cp37m-linux_x86_64.whl \ 
    && ~/miniconda3/bin/conda install pytorch torchvision cpuonly -c pytorch -n mindspore \
    && ~/miniconda3/envs/mindspore/bin/pip install -r ~/plato/requirements.txt \

