#!/bin/bash
# Run all benchmarks for all models with full samples.
# Pipeline models: allgaznie-glm, allgaznie-paddle, allgaznie-deepseek
# VLM-only models: glm-ocr, paddleocr-vl, deepseek-ocr2
#
# Datasets >200 samples are capped at 200 for time efficiency.
# Resume support: existing predictions are skipped automatically.

set -e
cd /root/ocr_test

MAX=""  # no cap — use full dataset
WARMUP=3
PORT=8000

echo "============================================="
echo " Full Benchmark Run - $(date)"
echo " Full dataset (no sample cap)"
echo "============================================="

# ─── Pipeline benchmarks ──────────────────────────
PIPELINE_BENCHES="omnidocbench upstage_dp_bench nanonets_kie"
PIPELINE_MODELS="allgaznie-glm allgaznie-paddle allgaznie-deepseek"

for model in $PIPELINE_MODELS; do
    echo ""
    echo ">>> Pipeline: $model"
    echo "─────────────────────────────────────"
    python3 infer.py --model "$model" \
        --benchmarks $PIPELINE_BENCHES \
        --warmup $WARMUP --port $PORT 2>&1
    echo ">>> $model pipeline done"
done

# ─── VLM-only benchmarks ─────────────────────────
VLM_BENCHES="ocrbench unimernet pubtabnet teds_test handwritten_forms"
VLM_MODELS="glm-ocr paddleocr-vl deepseek-ocr2"

for model in $VLM_MODELS; do
    echo ""
    echo ">>> VLM-only: $model"
    echo "─────────────────────────────────────"
    python3 infer.py --model "$model" \
        --benchmarks $VLM_BENCHES \
        --warmup $WARMUP --port $PORT 2>&1
    echo ">>> $model VLM-only done"
done

echo ""
echo "============================================="
echo " All inference complete - $(date)"
echo "============================================="

# ─── Run evaluation ──────────────────────────────
echo ""
echo ">>> Running evaluation for all models..."
python3 eval_bench.py --all 2>&1

echo ""
echo "============================================="
echo " All done - $(date)"
echo "============================================="
