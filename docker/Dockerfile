FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GITHASH=unknown

LABEL maintainer="Jacob Ramey rameyjm7@users.noreply.github.com"
LABEL org.opencontainers.image.title="ml-wireless-signal-classification"
LABEL org.opencontainers.image.revision="${GITHASH}"
LABEL org.opencontainers.image.description="RNN-LSTM model for wireless signal classification"

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        git \
        git-lfs \
        wget \
        curl \
        tar \
        build-essential \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10

ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
        tensorflow==2.16.1 \
        jupyterlab==4.2.5 \
        numpy \
        matplotlib \
        seaborn \
        scikit-learn \
        scipy \
        h5py \
        PyWavelets

WORKDIR /workspace

COPY code.tar.gz /workspace/code.tar.gz
RUN mkdir -p /workspace/code && \
    tar -xzf /workspace/code.tar.gz -C /workspace/code && \
    python3 -m pip install /workspace/code && \
    rm /workspace/code.tar.gz

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--notebook-dir=/workspace"]
