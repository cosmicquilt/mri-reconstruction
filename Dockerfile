# Reproducible CPU environment for the reconstruction pipeline.
# Build:  docker build -t mri-recon .
# Smoke:  docker run --rm mri-recon smoke
# Train:  docker run --rm -v ${PWD}/results:/app/results mri-recon train --config configs/synthetic.yaml
#
# For GPU training, start from an nvidia/cuda base image and install the matching
# CUDA build of torch (see https://pytorch.org/get-started/locally/).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# `mri-recon` is the console entry point (see pyproject.toml).
ENTRYPOINT ["mri-recon"]
CMD ["smoke"]
