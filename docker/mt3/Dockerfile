FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        libasound2-dev \
        libfluidsynth3 \
        libjack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN git clone --depth=1 https://github.com/google-research/t5x.git \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install gsutil nest-asyncio pyfluidsynth==1.3.0 \
    && python -m pip install -e /opt/t5x

RUN git clone --depth=1 https://github.com/magenta/mt3.git \
    && python -m pip install -e /opt/mt3

WORKDIR /workspace

COPY docker/mt3/mt3_runner.py /workspace/mt3_runner.py
COPY docker/mt3/entrypoint.sh /workspace/entrypoint.sh

RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
