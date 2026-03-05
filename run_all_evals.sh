#!/bin/bash
# Run all OmniDocBench evaluations with CDM fixes applied
# Each model uses a unique temp dir so results don't collide

set -e
cd /root/ocr_test

echo "============================================"
echo "Running ALL evaluations with CDM fixes"
echo "============================================"

# Models with OmniDocBench predictions that need re-eval
# (glm_ocr_pipeline and mineru_2.5 already evaluated post-fix)

echo ""
echo ">>> [1/4] GLM-OCR (VLM-only) OmniDocBench"
python3 eval_bench.py --model glm-ocr --benchmarks omnidocbench 2>&1

echo ""
echo ">>> [2/4] PaddleOCR-VL (VLM-only) OmniDocBench"
python3 eval_bench.py --model paddleocr-vl --benchmarks omnidocbench 2>&1

echo ""
echo ">>> [3/4] DeepSeek-OCR2 (VLM-only) OmniDocBench"
python3 eval_bench.py --model deepseek-ocr2 --benchmarks omnidocbench 2>&1

echo ""
echo ">>> [4/4] Allgaznie-GLM (Pipeline) OmniDocBench"
python3 eval_bench.py --model allgaznie-glm --benchmarks omnidocbench 2>&1

echo ""
echo "============================================"
echo "All OmniDocBench re-evaluations complete!"
echo "============================================"

# Now re-run non-OmniDocBench evaluations (these don't use CDM except unimernet)
# These should be fast since they don't require xelatex compilation
echo ""
echo ">>> Running all non-OmniDocBench evaluations..."

for model in glm-ocr paddleocr-vl deepseek-ocr2; do
    for bench in ocrbench pubtabnet teds_test unimernet handwritten_forms; do
        pred_dir="predictions/$(echo $model | tr '-' '_')/$bench"
        if [ -d "$pred_dir" ] && [ "$(find "$pred_dir" -type f | head -1)" ]; then
            echo ">>> $model / $bench"
            python3 eval_bench.py --model "$model" --benchmarks "$bench" 2>&1
        fi
    done
done

echo ""
echo "============================================"
echo "ALL evaluations complete!"
echo "============================================"
