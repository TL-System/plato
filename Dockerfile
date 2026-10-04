# Match the locked PyTorch CUDA toolkit. cuDNN comes from its Python wheel.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266
LABEL maintainer="Baochun Li"

COPY --from=ghcr.io/astral-sh/uv:0.12.22@sha256:f513a91fc62fe7c17567eee97230dd198e43edb8a9fbecca843714a4358fe1bc /uv /uvx /usr/local/bin/

# Keep import bytecode off the writable host checkout. User outputs still persist.
# Both the managed interpreter and project environment survive a checkout mount.
ENV UV_PYTHON_INSTALL_DIR=/opt/plato/python \
    UV_PROJECT_ENVIRONMENT=/opt/plato/.venv \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_NO_DEFAULT_GROUPS=1 \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/plato/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && uv --no-cache python install 3.13

WORKDIR /root/plato
# The complete retained workspace must exist before the locked editable sync.
COPY . /root/plato/
RUN uv sync --locked --python 3.13 --no-default-groups --no-cache \
    && sha256sum uv.lock > /opt/plato/lock.sha256

ARG PLATO_SOURCE_COMMIT=unknown
LABEL org.opencontainers.image.source="https://github.com/TL-System/plato" \
      org.opencontainers.image.revision="$PLATO_SOURCE_COMMIT"

CMD ["/bin/bash"]
