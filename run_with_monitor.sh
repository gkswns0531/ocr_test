#!/bin/bash
# Run benchmark with memory/CPU monitoring. Kill if memory > 85%.

THRESHOLD=85
CHECK_INTERVAL=5

cd /home/ubuntu/ocr_test

# Start benchmark in background
python main.py --models glm-ocr paddleocr-vl deepseek-ocr2 --benchmarks all --resume --port 8000 \
  > /home/ubuntu/ocr_test/benchmark.log 2>&1 &
BENCH_PID=$!

echo "[Monitor] Benchmark PID: $BENCH_PID"
echo "[Monitor] Memory threshold: ${THRESHOLD}%"
echo "[Monitor] Checking every ${CHECK_INTERVAL}s"

while kill -0 $BENCH_PID 2>/dev/null; do
    MEM_PCT=$(free | awk '/Mem:/ {printf "%.0f", $3/$2*100}')
    CPU_PCT=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d. -f1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "%.0f", $1/$2*100}')

    TIMESTAMP=$(date '+%H:%M:%S')
    echo "[$TIMESTAMP] RAM: ${MEM_PCT}% | CPU: ${CPU_PCT}% | GPU-MEM: ${GPU_MEM}%"

    if [ "$MEM_PCT" -ge "$THRESHOLD" ]; then
        echo "[$TIMESTAMP] *** RAM ${MEM_PCT}% >= ${THRESHOLD}% — KILLING BENCHMARK ***"
        kill $BENCH_PID 2>/dev/null
        sleep 2
        kill -9 $BENCH_PID 2>/dev/null
        # Also kill any vllm child processes and GPU orphans
        pkill -f "vllm.entrypoints" 2>/dev/null
        pkill -f "VLLM::EngineCore" 2>/dev/null
        nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | xargs -r kill -9 2>/dev/null
        echo "[$TIMESTAMP] Benchmark killed. Check benchmark.log for progress."
        exit 1
    fi

    sleep $CHECK_INTERVAL
done

echo "[Monitor] Benchmark finished normally (PID $BENCH_PID exited)."
echo "[Monitor] Results:"
ls -la /home/ubuntu/ocr_test/results/
