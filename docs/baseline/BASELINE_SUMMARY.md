# EyeAI — Gemessene Baseline (Vergleichsreferenz für Fixes)

**Datum**: 2026-07-26
**Gerät**: Samsung SM-S931B (Galaxy S25, NPU), ADB shell (kein Root)
**Paket**: `com.algorithmic_alliance.eyeaiapp.dev`
**Capture**: 8 s Warmup → 35 s gebounded Monitore (thermal/cpu/mem/logcat) + 30 s Perfetto-Trace
**Art der Last**: App gestartet (Launch-Activity `TutorialScreen`); Kamera/Audio/Depth-Threads bestätigt aktiv → repräsentativ für die laufende Pipeline.

> **Wichtige Einschränkung**: 35 s sind zu kurz für den Thermal-Throttle-Eintritt — Status blieb 0. Für den *Thermal*-Vergleichswert ist ein längerer Soak (3–5 min) nötig (siehe unten). Die CPU-/Mem-Werte sind aber bereits aussagekräftig.

## Gemessene Werte

### CPU (Peak über 35 Snapshots)
| Metrik | Wert |
|---|---|
| Peak gesamt | **430 %user + 219 %sys = ~649 / 800 % (~81 % aller 8 Kerne)** |
| Idle (meist) | < 30 % → nahezu alle Kerne dauerhaft beschäftigt |
| Bereich user | 300–430 % |

**Top-Threads (letzter Snapshot, %CPU):**
| %CPU | TID | Thread | Interpretation → Finding |
|---|---|---|---|
| 82.7 | 11158 | `pool-6-thread-1` | einer der Inference-Loops → **#4 doppelte Inferenz** |
| 51.7 | 10073 | `pool-10-thread-1` | zweiter Inference-Loop → **#4** |
| 24.1 | 9997 | `HeapTaskDaemon` | **GC-Druck** (kleine per-Frame-Allocs: Matrix/RectF, String.format, colormap IntArray) |
| 17.2 | 10070 | `pool-9-thread-1` | Kamera-Analyzer-/Audio-Pool |
| 17.2 | 9991 | `…eyeaiapp.dev` (Main) | UI-Hop je Frame → **#ui-thread-hop** |
| 13.7 | 10098 | `pool-13-thread-1` | Executor-Pool |
| 10.3 | 11156 | `CameraX-camerax` | Sensor/ISP + analyze() → **#2 Kamera-FPS** |
| 6.8 | 10336 | `AAudio_1` | Audio-Ausgabe → **#3/#4 Audio** |
| 6.8 | 10034 | `RenderThread` | depthView/Overlay-Draw je Frame → **#6 Debug-Render** |

### Thermal
- `Thermal Status: 0` durchgehend (35 s, Kaltstart). → **keine Aussage über Throttle-Kurve**; längerer Soak nötig.

### Memory
| Metrik | Wert |
|---|---|
| Native Heap (first → last) | 388 364 kB → 401 380 kB = **+13 016 kB in 35 s** |
| TOTAL PSS | ~843 MB |
| RES | 768 MB |
| VIRT | 19 GB (nur Adressraum, ML-Runtime-Reservierung) |
| Swap used | 4,5 GB (System-Swap aktiv — Zeichen für Memory-Pressure) |

→ Native-Heap-Wachstum bestätigt **#8 managed_host-Churn** (und/oder QNN-interne Puffer).

### Logcat
- "Native Lib"-Tag: **0 Zeilen** in 35 s (Debug-Build). → **#10 always-on Logging** in *Release/Production* separat prüfen.

### Perfetto-Trace
- `perfetto_baseline.perfetto-trace`, **33,47 MB**, valides Proto (Header `0a 0a 08 06 10 …`).
- Data Sources enthalten: `linux.ftrace`, `linux.process_stats`, `linux.system_info`, `binder_driver`, `android.surfaceflinger.frametimeline`, sched/freq/camera/gfx/…
- **Öffnen**: Datei nach `https://ui.perfetto.dev` ziehen. Erwartet: zwei überlappende `pool-*`-Inference-Threads, `CameraX`-hohe Sensor-Frequenz, `AAudio`-Wakeups, Main-`gfx`-Spikes je Frame.

## Mapping: Messwert → Finding (Bestätigung der Analyse)
- **#4 doppelte Inferenz / HTP-Contention**: `pool-6` 82,7 % + `pool-10` 51,7 % = zwei Inference-Loops gleichzeitig. ✓
- **GC**: `HeapTaskDaemon` 24 % — Bitmap/FloatBuffer ist reuse-t, aber kleine per-Frame-Allocs erzeugen noch GC-Last. ✓ (nuanciert: nicht der Hauptfresser, aber sichtbar)
- **#8 managed_host**: Native Heap +13 MB/35 s Wachstum. ✓
- **#2 Kamera + #6 Debug-Render**: `CameraX-camerax` 10,3 % + `RenderThread` 6,8 % + Main 17,2 %. ✓
- **#3/#4 Audio**: `AAudio_1` 6,8 % sichtbar (Rust-Audio-Threads tauchen unter `pool-*`/eigenen TIDs auf — im Perfetto `sched`-Track auflösbar). ✓

## 5-Min-Thermal-Soak (gemessen — HAUPT-Vergleichswert)

`run_thermal_soak.sh` — 120 s Cool-down (App gestoppt) → Kalt-start → 300 s Last @ 1 Hz. Sensoren via `dumpsys thermalservice` (kein Root nötig). Artefakte: `thermal_soak_5min.csv`, `thermal_soak_raw.txt`, `cpu_soak_5min.txt`.

**Throttle-Verlauf (der Beweis):**
| t | Thermal Status | Bedeutung |
|---|---|---|
| 0–3 s | 1 (3 s) | light |
| 4–130 s | **2 (127 s)** | moderate |
| 131–300 s | **3 (170 s)** | **severe** — ab t≈131 s, bis Stop |

→ **Die App treibt den S25 binnen ~2 min in SEVERE-Throttling (Status 3) und kommt nicht mehr zurück auf 0.** Nie im 5-Min-Fenster auf 0. Das ist das gemessene Äquivalent zu „nach kurzer Zeit thermisch drosselt".

**Temperaturen:**
| Sensor | Start | Peak | Ende | Δ |
|---|---|---|---|---|
| AP (SoC-Die) | 48,5 °C | 65,3 °C (t=4 s, Pre-Throttle-Spike) | 56,4 °C | Spike→Drossel senkt Die-Temp |
| SKIN | 38,2 °C | 46,6 °C (t=283 s, am Ende) | 46,6 °C | **+8,4 °C, stetig, noch steigend** |
| Batterie | 38,8 °C | 46,5 °C (t=295 s, am Ende) | 46,5 °C | **+7,7 °C, noch steigend bei Stop** |

- SKIN ist der Throttle-Treiber (`skin_status` 1→3; `ap_status` blieb 0). SKIN/Batterie zeigen **keine Plateaubildung** — bei längerer Laufzeit wäre Status 4 (critical) zu erwarten.
- Der AP-Spike 48,5→65,3 °C in 4 s und das anschließende Absinken auf 56,4 °C ist die **Drosselung selbst** (Frequenz-Cap → weniger Leistung → Die kühlt leicht, aber Status bleibt 3).

**CPU unter Drosselung (kein Backoff — Smoking Gun für Finding #1):**
| t | pool-6 (Inf.) | pool-10 (Inf.) | HeapTaskDaemon | pool-13 |
|---|---|---|---|---|
| 15 s (heiß) | 86,2 % | 48,2 % | 24,1 % | 31,0 % |
| 120 s | 82,7 % | 44,8 % | 24,1 % | 27,5 % |
| 300 s (status 3) | 82,7 % | 41,3 % | 24,1 % | 24,1 % |

→ `pool-6-thread-1` bleibt durchgehend bei **82–86 %**, auch unter severe Throttling. **Die App reduziert ihre Last bei Hitze nicht** — sie hämmert weiter, das Gerät bleibt bei Status 3. Genau das adressiert Finding #1 (Thermal-Feedback-Loop): bei Status≥2 müssten FPS/Auflösung sinken, statt weiter mit 30+ fps zu inferieren.

**Vergleichs-Ziele nach Fixes (PR 1: Thermal-Loop + Kamera-FPS + Debug-Gate + Profiling):**
- Time-to-status-2: > 60 s (statt 4 s); Time-to-status-3: gar nicht erst erreichen im 5-Min-Fenster.
- SKIN-Δ über 5 min: < +3 °C (statt +8,4 °C); Batterie-Δ < +3 °C.
- `pool-6` unter Last: < 50 % (statt 82–86 %); `HeapTaskDaemon` < 10 %.
- Unter Status≥2: CPU-Backoff sichtbar (Loop-Pegel sinkt) — derzeit nicht der Fall.

## Vergleichs-Workflow nach Fixes
1. Dieselbe `run_baseline.sh` (bzw. Soak) auf dem *gefixten* Build laufen lassen.
2. Vergleichen: Peak-%CPU (Ziel: < ~40 %), `HeapTaskDaemon` %, Native-Heap-Wachstum (Ziel: flach), Thermal-Status-Übergang (Ziel: bleibt länger auf 0 / steigt langsamer).
3. Perfetto vor/nach in ui.perfetto.dev vergleichen (Inference-Thread-Überlappung, Audio-Wakeups, `gfx`-Spikes).
