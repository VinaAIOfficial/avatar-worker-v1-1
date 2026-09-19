FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install pip==24.0 setuptools==69.5.1 wheel==0.43.0

# Pin the upstream code inspected for this starter; do not track main silently.
ARG SADTALKER_COMMIT=cd4c0465ae0b54a6f85af57f5c65fec9fe23e7f8
RUN git clone https://github.com/OpenTalker/SadTalker.git /opt/SadTalker \
    && cd /opt/SadTalker && git checkout "$SADTALKER_COMMIT" \
    && git rev-parse HEAD > /opt/sadtalker-version.txt && rm -rf .git
WORKDIR /opt/SadTalker
COPY constraints.txt /opt/worker/constraints.txt
RUN python3 -m pip install -c /opt/worker/constraints.txt torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
COPY requirements-worker.txt /opt/worker/requirements-worker.txt
RUN python3 -m pip install --no-build-isolation \
    -c /opt/worker/constraints.txt -r /opt/worker/requirements-worker.txt
COPY download_models.py /opt/worker/download_models.py
RUN python3 /opt/worker/download_models.py
# Import smoke test requires no GPU; actual inference is tested after deployment.
RUN python3 -c "import runpod, torch, cv2; from src.utils.preprocess import CropAndExtract; from src.test_audio2coeff import Audio2Coeff; from src.facerender.animate import AnimateFromCoeff"
COPY handler.py /opt/worker/handler.py
COPY references/ /opt/worker/references/
COPY test_input.json /opt/SadTalker/test_input.json
ENV SADTALKER_DIR=/opt/SadTalker
CMD ["python3", "-u", "/opt/worker/handler.py"]
