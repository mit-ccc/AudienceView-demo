#!/bin/bash

# Set up the packages needed for all images in a conda environment, for testing

set -Eeuo pipefail
set -x

PYTHON_VERSION=3.10
CUDA_VERSION=12.1.0
PYTORCH_VERSION=2.2.2
ENV_NAME='audienceview'

"$HOME/miniconda3/bin/conda" create -y -n "$ENV_NAME" -c defaults \
    python="$PYTHON_VERSION" ipykernel ipywidgets widgetsnbextension ipython

"$HOME/miniconda3/bin/mamba" install -y -n "$ENV_NAME" -c "nvidia/label/cuda-$CUDA_VERSION" \
    cuda="$CUDA_VERSION"

"$HOME/miniconda3/bin/mamba" install -y -n "$ENV_NAME" -c pytorch -c nvidia -c defaults \
    pytorch="$PYTORCH_VERSION" torchtext torchdata torchtriton pytorch-cuda

"$HOME/miniconda3/bin/mamba" install -y -n "$ENV_NAME" -c conda-forge \
    altair google-api-python-client hdbscan matplotlib nltk numpy openai \
    pandas scipy sqlalchemy statsmodels streamlit tqdm transformers \
    umap-learn wordcloud
