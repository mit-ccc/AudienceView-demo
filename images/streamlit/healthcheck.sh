#!/bin/bash

set -Eeuo pipefail

if [ "$MODE" == "refresh" ]; then
    exit 0
fi

curl --fail http://localhost:8501/_stcore/health || exit 1
