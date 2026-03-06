#!/usr/bin/env bash
# 3-way MinerU comparison benchmark
# MinerU Original results already exist (3,279ms, 90.6 Overall)
set -euo pipefail

cd /root/ocr_test

echo "=== Phase 1: MinerU Optimized ==="
python3 infer.py --model mineru-optimized --benchmarks omnidocbench --warmup 3

echo ""
echo "=== Phase 2: Allgaznie-MinerU ==="
python3 infer.py --model allgaznie-mineru --benchmarks omnidocbench --warmup 3

echo ""
echo "=== Phase 3: Evaluation ==="
python3 eval_bench.py --all

echo ""
echo "=== Done ==="
