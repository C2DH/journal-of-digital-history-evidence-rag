#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
# CPU-only llama-cpp-python build. If a wheel is available, pip will use it.
CMAKE_ARGS="-DGGML_CUDA=OFF -DGGML_METAL=OFF" python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
python -m pip install -r requirements.txt
cp -n .env.example .env || true
echo "Installation complete. Activate with: source .venv/bin/activate"
