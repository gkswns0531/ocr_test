#!/bin/bash
# Run benchmark with memory watchdog (kills if RAM > 90%)
set -e

THRESHOLD=90
CMD="python main.py --models $@ --benchmarks ocrbench handwritten_forms nanonets_kie --resume"

echo "=== Starting: $CMD ==="
echo "=== Memory watchdog: kill if RAM > ${THRESHOLD}% ==="

$CMD &
PID=$!

while kill -0 $PID 2>/dev/null; do
    MEM_PCT=$(free | awk '/Mem:/ {printf "%.0f", $3/$2*100}')
    echo "[watchdog] RAM: ${MEM_PCT}%  $(date +%H:%M:%S)"
    if [ "$MEM_PCT" -gt "$THRESHOLD" ]; then
        echo "[watchdog] RAM ${MEM_PCT}% > ${THRESHOLD}%! KILLING ALL!"
        kill -9 $PID 2>/dev/null
        pkill -9 -f "vllm" 2>/dev/null
        echo "[watchdog] Killed. Exiting."
        exit 1
    fi
    sleep 5
done

wait $PID
EXIT_CODE=$?
echo "=== Benchmark finished with exit code $EXIT_CODE ==="
exit $EXIT_CODE
