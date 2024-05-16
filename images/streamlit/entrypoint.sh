#!/bin/bash

set -Eeuo pipefail

if [ "$MODE" == "refresh" ]; then
    ./refresh.py
elif [ "$MODE" == "run" ]; then
    exec streamlit run main.py --server.port=8504 --server.address=0.0.0.0
else
    echo "Error: Invalid MODE value. Please set MODE to 'refresh' or 'run'."
    exit 1
fi
