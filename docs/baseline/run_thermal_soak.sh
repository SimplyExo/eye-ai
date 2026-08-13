#!/usr/bin/env bash
# EyeAI 5-minute Thermal Soak (CORRECTED: real 1Hz, AP/SKIN sensor temps, no root).
# Cool-down first (app stopped) for a cold-ish start, then 300s load soak.
set -u
OUT="$(cd "$(dirname "$0")" && pwd)"; cd "$OUT"
PKG=com.algorithmic_alliance.eyeaiapp.dev
ACT=com.algorithmic_alliance.eyeaiapp.dev/com.algorithmic_alliance.eyeaiapp.TutorialScreen
COOLDOWN="${COOLDOWN:-120}"
DURATION=300

get_temps() {  # prints: status ap apst skin skinst bat
  adb shell "dumpsys thermalservice | grep -E 'Thermal Status:|mName=AP|mName=SKIN'; dumpsys battery | grep -m1 '^  temperature:'" 2>/dev/null | tr -d '\r'
}

echo "=== Cool-down: stop app, idle ${COOLDOWN}s ==="
adb shell am force-stop "$PKG"
START_AP="$(get_temps | grep 'mName=AP' | tail -1 | grep -oE 'mValue=[0-9.]+' | cut -d= -f2)"
echo "AP temp at cool-down start: ${START_AP:-?} C"
sleep "$COOLDOWN"

echo "=== Wake + relaunch + warmup 8s ==="
adb shell input keyevent 224 >/dev/null 2>&1
sleep 1
adb shell am start -n "$ACT" >/dev/null 2>&1
sleep 8

: > thermal_soak_raw.txt
: > cpu_soak_5min.txt

echo "=== Soak start: $(date '+%H:%M:%S') — ${DURATION}s @ 1Hz ==="
for i in $(seq 1 "$DURATION"); do
  {
    echo "[t=$(printf '%03d' "$i")]"
    get_temps
  } >> thermal_soak_raw.txt
  if (( i % 15 == 0 )); then
    {
      echo "--- t=${i}s $(date '+%H:%M:%S') ---"
      adb shell top -H -n 1 -p "$(adb shell pidof "$PKG" | tr -d '\r')" 2>/dev/null \
        | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | grep -E '^[[:space:]]*[0-9]+ u0_a498' | head -8
    } >> cpu_soak_5min.txt
    printf '  [cpu snapshot @ t=%ss captured]\n' "$i"
  fi
  sleep 1
done
echo "=== Soak end: $(date '+%H:%M:%S') ==="

echo ""
echo "=== Parsing ==="
python3 - <<'PY'
import re
blocks=open('thermal_soak_raw.txt').read().split('[t=')
rows=[]
for b in blocks:
    if not b.strip(): continue
    m=re.match(r'(\d+)\]',b); t=int(m.group(1)) if m else None
    if t is None: continue
    status=None; ap=None; apst=None; skin=None; skinst=None; bat=None
    for ln in b.splitlines():
        if 'Thermal Status:' in ln:
            mm=re.search(r'Thermal Status:\s*(\d+)',ln); status=int(mm.group(1)) if mm else None
        elif 'mName=AP' in ln:
            v=re.search(r'mValue=([0-9.]+)',ln); s=re.search(r'mStatus=(\d+)',ln)
            if v: ap=float(v.group(1))   # last occurrence wins (HAL/current)
            if s: apst=int(s.group(1))
        elif 'mName=SKIN' in ln:
            v=re.search(r'mValue=([0-9.]+)',ln); s=re.search(r'mStatus=(\d+)',ln)
            if v: skin=float(v.group(1))
            if s: skinst=int(s.group(1))
        elif ln.strip().startswith('temperature:'):
            mm=re.search(r'temperature:\s*(\d+)',ln); bat=int(mm.group(1)) if mm else None
    rows.append((t,status,ap,apst,skin,skinst,bat))

with open('thermal_soak_5min.csv','w') as f:
    f.write('t,thermal_status,ap_c,ap_status,skin_c,skin_status,battery_c\n')
    for r in rows: f.write('%d,%s,%s,%s,%s,%s,%s\n'%r)

def col(i): return [r[i] for r in rows if r[i] is not None]
print("samples:",len(rows))
# status histogram
from collections import Counter
hc=Counter(r[1] for r in rows if r[1] is not None)
print("thermal_status histogram (seconds per level):",dict(sorted(hc.items())))
def first_status(level):
    for r in rows:
        if r[1] is not None and r[1]>=level: return r[0]
    return None
print("time-to-status>=1:",first_status(1),"s")
print("time-to-status>=2:",first_status(2),"s")
print("time-to-status>=3:",first_status(3),"s")
def stats(name,idx):
    v=col(idx)
    if not v: print(f"{name}: no data"); return
    print(f"{name}: start={v[0]} end={v[-1]} peak={max(v)} (at t={rows[[r[idx] for r in rows].index(max(v))][0]}s) min={min(v)}")
stats("AP(SoC) C",2)
stats("SKIN C",4)
stats("Battery C (tenths)",6)
PY
echo ""
echo "=== Artifacts ==="
wc -l thermal_soak_raw.txt cpu_soak_5min.txt; ls -la thermal_soak_5min.csv