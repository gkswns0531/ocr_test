#!/usr/bin/env bash
# setup_b200.sh — B200 환경 설치 스크립트
# Usage: bash setup_b200.sh
set -euo pipefail

echo "=== B200 OCR+Embedding Pipeline Setup ==="

# 1. System packages
echo "[1/7] System packages..."
apt-get update -qq && apt-get install -y -qq python3-pip git

# 2. PyTorch + CUDA 12.4
echo "[2/7] PyTorch + CUDA..."
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. vLLM (B200 / Blackwell support)
echo "[3/7] vLLM..."
pip install --quiet vllm

# 4. Transformers (latest dev for PP-DocLayoutV3 support)
echo "[4/7] Transformers (dev)..."
pip install --quiet git+https://github.com/huggingface/transformers.git

# 5. GLM-OCR SDK
echo "[5/7] GLM-OCR SDK..."
if [ ! -d "/home/ubuntu/glm-ocr-sdk" ]; then
    git clone https://github.com/zai-org/glm-ocr.git /home/ubuntu/glm-ocr-sdk
fi
cd /home/ubuntu/glm-ocr-sdk && pip install --quiet -e .
cd -

# 6. Additional dependencies
echo "[6/7] Python dependencies..."
pip install --quiet \
    datasets \
    huggingface_hub \
    lxml \
    qwen-vl-utils \
    numpy \
    Pillow \
    tqdm \
    pyyaml

# 7. HF login (interactive)
echo "[7/7] HuggingFace login..."
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Run: huggingface-cli login"
    echo "Or set HF_TOKEN environment variable before running the pipeline."
else
    huggingface-cli login --token "$HF_TOKEN"
    echo "Logged in with HF_TOKEN."
fi

echo ""
echo "=== Setup complete ==="
echo "Next: python3 run_b200_pipeline.py --hf-repo YOUR_USERNAME/sds-kopub-ocr-embeddings --hf-token hf_xxxxx"
