#!/bin/bash
# Focused benchmark: GLM-OCR ecosystem + MinerU only
cd /root/ocr_test

LOG="/root/ocr_test/benchmark_focused.log"
TIMELOG="/root/ocr_test/benchmark_timing_focused.log"

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
    return 0
}

echo "phase,elapsed_s,exit_code" > "$TIMELOG"
# Keep glm-ocr timing from previous run
echo "glm-ocr_vlm_benchmarks,4058,0" >> "$TIMELOG"

log "============================================"
log "FOCUSED BENCHMARK: GLM + MinerU"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
log "============================================"

# Phase 1: GLM-OCR Pipeline (official SDK)
time_phase "glm-ocr-pipeline_doc_benchmarks" \
    python3 infer.py --model glm-ocr-pipeline \
    --benchmarks omnidocbench upstage_dp_bench nanonets_kie \
    --warmup 3

# Phase 2: Allgaznie-GLM (our pipeline)
time_phase "allgaznie-glm_doc_benchmarks" \
    python3 infer.py --model allgaznie-glm \
    --benchmarks omnidocbench upstage_dp_bench nanonets_kie \
    --warmup 3

# Phase 3: MinerU
time_phase "mineru_omnidocbench" \
    python3 infer.py --model mineru --benchmarks omnidocbench

# Phase 4: Evaluation for all models with predictions
log "============================================"
log "Phase 4: Running evaluations"
log "============================================"

time_phase "eval_all" \
    python3 eval_bench.py --all

log "============================================"
log "FOCUSED BENCHMARK COMPLETE"
log "============================================"
log "Timing summary:"
cat "$TIMELOG" | tee -a "$LOG"
