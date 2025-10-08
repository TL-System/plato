# Download base image from NVIDIA's Docker Hub
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04
LABEL maintainer="Baochun Li"

WORKDIR /root/plato
COPY . /root/plato/

RUN apt-get update \
    && apt-get install -y libgomp1 \
    && apt-get install -y curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Set environment variables for uv before using it
ENV PATH="/root/.local/bin:$PATH"

# Set additional environment variables for uv
ENV UV_SYSTEM_PYTHON=1

# Set default command to use uv run
CMD ["/bin/bash"]
