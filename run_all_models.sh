#!/bin/bash
set -e

echo "========================================"
echo "1/4: GLM-OCR-Pipeline (문서 벤치마크)"
echo "========================================"
python3 -u infer.py --model glm-ocr-pipeline --benchmarks omnidocbench upstage_dp_bench nanonets_kie --port 8000

echo "========================================"
echo "2/4: GLM-OCR VLM (크롭 벤치마크)"
echo "========================================"
python3 -u infer.py --model glm-ocr --benchmarks ocrbench unimernet pubtabnet teds_test handwritten_forms --port 8000

echo "========================================"
echo "3/4: PaddleOCR-VL-Pipeline (문서 벤치마크)"
echo "========================================"
python3 -u infer.py --model paddleocr-vl-pipeline --benchmarks omnidocbench upstage_dp_bench nanonets_kie --port 8000

echo "========================================"
echo "4/4: PaddleOCR-VL VLM (크롭 벤치마크)"
echo "========================================"
python3 -u infer.py --model paddleocr-vl --benchmarks ocrbench unimernet pubtabnet teds_test handwritten_forms --port 8000

echo "========================================"
echo "전체 평가"
echo "========================================"
python3 -u eval_bench.py --all

echo "완료!"
