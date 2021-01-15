# Download base image ubuntu 20.04
FROM ubuntu:20.04
LABEL maintainer="Baochun Li"

WORKDIR /root

ADD .bashrc /root/

RUN apt-get update \
    && apt-get install -y wget \
    && apt-get install -y vim \
    && apt-get install -y git \
    && mkdir -p ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm -rf ~/miniconda3/miniconda.sh \
    && ~/miniconda3/bin/conda init bash \
    && ~/miniconda3/bin/conda create -n federated python=3.8 \
    && ~/miniconda3/bin/conda install pytorch torchvision -c pytorch -n federated \
    && ~/miniconda3/bin/conda install matplotlib pylint scipy -n federated \
    && ~/miniconda3/bin/conda install -c conda-forge websockets -n federated \
    && ~/miniconda3/bin/conda create -n mindspore python=3.7 \
    && ~/miniconda3/envs/mindspore/bin/pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.1.0/MindSpore/cpu/ubuntu_x86/mindspore-1.1.0-cp37-cp37m-linux_x86_64.whl \
    && ~/miniconda3/bin/conda install matplotlib pylint -n mindspore \
    && ~/miniconda3/bin/conda install -c conda-forge websockets -n mindspore

