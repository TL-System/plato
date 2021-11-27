# Download base image from NVIDIA's Docker Hub
FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
LABEL maintainer="Baochun Li"

ADD ./.bashrc /root/
COPY ./requirements.txt /root/
WORKDIR /root/plato

RUN apt-get update \
    && apt-get install -y wget \
    && apt-get install -y vim \
    && apt-get install -y net-tools \
    && apt-get install -y git \
    && apt-get install -y libgmp-dev \
    && mkdir -p ~/miniconda3 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh \
    && bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 \
    && rm -rf ~/miniconda3/miniconda.sh \
    && ~/miniconda3/bin/conda update -n base -c defaults conda \
    && ~/miniconda3/bin/conda init bash \
    && ~/miniconda3/bin/conda create -n plato python=3.9 \
    && ~/miniconda3/bin/conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -n plato -y \
    && ~/miniconda3/envs/plato/bin/pip install -r ~/requirements.txt \
    && ~/miniconda3/envs/plato/bin/pip install plato-learn

RUN rm /root/requirements.txt
