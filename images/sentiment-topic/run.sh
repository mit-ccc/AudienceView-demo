#!/bin/bash

set -Eeuo pipefail

python3 sentiment.py

python3 topic-embeds.py
python3 topic-umap.py
python3 topic-hdbscan-train.py
python3 topic-hdbscan-predict.py
