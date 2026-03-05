#!/bin/bash
# Full OCR benchmark: wipe all results, re-run all inference + evaluation
# Captures latency per model×benchmark via _latency.json files
#
# Model×Benchmark Matrix:
#   VLM-only (glm-ocr, paddleocr-vl, deepseek-ocr2):
#     → omnidocbench, ocrbench, pubtabnet, teds_test, unimernet, handwritten_forms
#   Pipeline (glm-ocr-pipeline, mineru):
#     → omnidocbench, upstage_dp_bench, nanonets_kie
#   Allgaznie (allgaznie-glm, allgaznie-paddle, allgaznie-deepseek):
#     → omnidocbench, upstage_dp_bench, nanonets_kie
#
# paddleocr-vl-pipeline is SKIPPED (PaddlePaddle 2.6.2 segfaults on Blackwell sm_120)

cd /root/ocr_test

LOG="/root/ocr_test/benchmark_full.log"
TIMELOG="/root/ocr_test/benchmark_timing.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

time_phase() {
    local phase="$1"
    shift
    local t0=$(date +%s)
    log "START: $phase"
    "$@" 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    local t1=$(date +%s)
    local elapsed=$((t1 - t0))
    echo "$phase,$elapsed,$rc" >> "$TIMELOG"
    if [ $rc -ne 0 ]; then
        log "FAILED: $phase (exit=$rc, ${elapsed}s)"
    else
        log "DONE: $phase (${elapsed}s)"
    fi
    return 0  # always continue
}

# ============================================================
# Phase 0: Clear everything
# ============================================================
log "============================================"
log "FULL BENCHMARK RE-RUN STARTING"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
log "============================================"

echo "phase,elapsed_s,exit_code" > "$TIMELOG"

log "Clearing all predictions and results..."
rm -rf /root/ocr_test/predictions/*
rm -rf /root/ocr_test/results/*
# Also clear OmniDocBench cached results
rm -rf /root/OmniDocBench/result/
# Clear CDM cache to ensure fresh evaluation
find /root/OmniDocBench -name "CDM" -type d -exec rm -rf {} + 2>/dev/null || true
log "All cleared."

# ============================================================
# Phase 1: VLM-only models
# ============================================================

# 1a. GLM-OCR (vLLM server, ~4-5 hours total)
time_phase "glm-ocr_vlm_benchmarks" \
    python3 infer.py --model glm-ocr \
    --benchmarks ocrbench pubtabnet teds_test unimernet handwritten_forms omnidocbench \
    --warmup 3

# 1b. PaddleOCR-VL (vLLM server, ~4-5 hours)
time_phase "paddleocr-vl_vlm_benchmarks" \
    python3 infer.py --model paddleocr-vl \
    --benchmarks ocrbench pubtabnet teds_test unimernet handwritten_forms omnidocbench \
    --warmup 3

# 1c. DeepSeek-OCR2 (offline vLLM, ~3-4 hours)
time_phase "deepseek-ocr2_vlm_benchmarks" \
    python3 infer.py --model deepseek-ocr2 \
    --benchmarks ocrbench pubtabnet teds_test unimernet handwritten_forms omnidocbench \
    --warmup 3

# ============================================================
# Phase 2: Official Pipeline models
# ============================================================

# 2a. GLM-OCR Pipeline (vLLM + glmocr SDK, ~5-7 hours)
time_phase "glm-ocr-pipeline_doc_benchmarks" \
    python3 infer.py --model glm-ocr-pipeline \
    --benchmarks omnidocbench upstage_dp_bench nanonets_kie \
    --warmup 3

# 2b. MinerU (local pipeline, ~5-8 hours)
time_phase "mineru_omnidocbench" \
    python3 infer.py --model mineru --benchmarks omnidocbench

# ============================================================
# Phase 3: Allgaznie Pipeline models
# ============================================================

# 3a. Allgaznie-GLM (vLLM fp8 + local layout model)
time_phase "allgaznie-glm_doc_benchmarks" \
    python3 infer.py --model allgaznie-glm \
    --benchmarks omnidocbench upstage_dp_bench nanonets_kie \
    --warmup 3

# 3b. Allgaznie-Paddle (vLLM + local layout model)
time_phase "allgaznie-paddle_doc_benchmarks" \
    python3 infer.py --model allgaznie-paddle \
    --benchmarks omnidocbench upstage_dp_bench nanonets_kie \
    --warmup 3

# 3c. Allgaznie-DeepSeek (vLLM fp8 + local layout model)
time_phase "allgaznie-deepseek_doc_benchmarks" \
    python3 infer.py --model allgaznie-deepseek \
    --benchmarks omnidocbench upstage_dp_bench nanonets_kie \
    --warmup 3

# ============================================================
# Phase 4: Evaluation
# ============================================================
log "============================================"
log "Phase 4: Running ALL evaluations"
log "============================================"

time_phase "eval_all" \
    python3 eval_bench.py --all

# ============================================================
# Phase 5: Summary
# ============================================================
log "============================================"
log "FULL BENCHMARK COMPLETE"
log "============================================"
log "Timing summary:"
cat "$TIMELOG" | tee -a "$LOG"

echo ""
echo "=== DONE ==="
echo "Results: /root/ocr_test/results/"
echo "Latency: /root/ocr_test/predictions/*/_latency.json"
echo "Timing: $TIMELOG"
echo "Full log: $LOG"
