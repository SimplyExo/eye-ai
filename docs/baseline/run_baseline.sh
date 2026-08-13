#!/usr/bin/env bash
# EyeAI Thermal/Performance Baseline Measurement
# Self-terminating monitors + 30s Perfetto trace. Safe to re-run.
set -u
OUT="$(cd "$(dirname "$0")" && pwd)"
cd "$OUT"

echo "=== Detecting installed eyeai package ==="
PKG="$(adb shell pm list packages 2>/dev/null | grep -i eyeai | head -1 | sed 's/package://' | tr -d '\r')"
if [ -z "$PKG" ]; then echo "ERROR: no eyeai package found. Is the app installed?"; exit 1; fi
echo "PKG=$PKG"

echo "=== Resolving launch activity ==="
ACT="$(adb shell cmd package resolve-activity --brief "$PKG" 2>/dev/null | tail -1 | tr -d '\r')"
echo "ACT=$ACT"

echo "=== Perfetto availability ==="
adb shell "command -v perfetto" 2>&1 | head -1

echo "=== Force-stop + launch + warmup (8s) ==="
adb shell am force-stop "$PKG"
adb shell am start -n "$ACT" >/dev/null 2>&1 || adb shell monkey -p "$PKG" -c android.intent.category.LAUNCHER 1 >/dev/null 2>&1
sleep 8

DURATION=35   # monitor window (s) — covers the 30s perfetto trace
echo "=== Starting bounded monitors for ${DURATION}s -> *.txt ==="
# Thermal status (no root) — self-terminating
( for i in $(seq 1 "$DURATION"); do
    echo "--- t=${i}s $(date +%H:%M:%S) ---"
    adb shell dumpsys thermalservice 2>/dev/null | grep -iE "status|throttl" | head -3
  done ) > thermal_baseline.txt &
THERMAL_PID=$!

# CPU per thread (top -H) — self-terminating
( for i in $(seq 1 "$DURATION"); do
    echo "--- t=${i}s $(date +%H:%M:%S) ---"
    adb shell top -H -n 1 -p "$(adb shell pidof "$PKG" | tr -d '\r')" 2>/dev/null | head -22
  done ) > cpu_baseline.txt &
CPU_PID=$!

# Memory / GC / native heap — self-terminating, every 2s
( for i in $(seq 1 $((DURATION/2))); do
    echo "--- t=$((i*2))s $(date +%H:%M:%S) ---"
    adb shell dumpsys meminfo "$PKG" 2>/dev/null | grep -iE "Native Heap|Dalvik|TOTAL|GC:"
    sleep 2
  done ) > mem_baseline.txt &
MEM_PID=$!

# Logcat volume proxy (Native Lib trace/debug in release) — count lines over window
( adb logcat -c 2>/dev/null
  for i in $(seq 1 "$DURATION"); do sleep 1; done
  adb logcat -d -s "Native Lib:V" 2>/dev/null | wc -l ) > logcat_native_count.txt &
LOG_PID=$!

echo "=== Starting 30s Perfetto trace ==="
TRACE=device_trace.perfetto-trace
adb shell perfetto -o "/data/local/tmp/$TRACE" -t 30s \
  sched freq idle am wm gfx view binder_driver hal dalvik camera input res memory 2>&1 | tail -3
PERFETTO_RC=${PIPESTATUS[0]}

echo "=== Waiting for monitors to finish ==="
wait $THERMAL_PID $CPU_PID $MEM_PID $LOG_PID 2>/dev/null

echo "=== Pulling Perfetto trace ==="
if adb shell "test -f /data/local/tmp/$TRACE" 2>/dev/null; then
  adb pull "/data/local/tmp/$TRACE" ./perfetto_baseline.perfetto-trace 2>&1 | tail -1
  adb shell "rm /data/local/tmp/$TRACE" 2>/dev/null
else
  echo "WARN: no perfetto trace on device (perfetto missing or failed, rc=$PERFETTO_RC)"
fi

echo ""
echo "=== DONE. Artifacts in $OUT ==="
ls -la "$OUT"
echo ""
echo "Thermal samples: $(wc -l < thermal_baseline.txt) lines"
echo "CPU samples:      $(wc -l < cpu_baseline.txt) lines"
echo "Mem samples:      $(wc -l < mem_baseline.txt) lines"
echo "Native logcat lines over ${DURATION}s: $(cat logcat_native_count.txt 2>/dev/null)"
