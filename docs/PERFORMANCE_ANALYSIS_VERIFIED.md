# EyeAI — Verifizierte Thermal-/Performance-Analyse (Arbeitsreferenz)

**Datum**: 2026-07-26
**Methode**: 20-Agent-Workflow (10 Subsystem-Deep-Reader → adversarielle Verifikation der Top-9-Claims → Completeness-Critic), plus first-hand Code-Verifikation der überraschendsten Befunde.
**Status**: NUR Analyse. Keine Code-Änderungen.
**Ablösung**: Diese Datei korrigiert `docs/PERFORMANCE_ANALYSIS.md` (vorheriger Entwurf, enthält nachweisbare Fehler — siehe §6).

> **Kern-Erkenntnis vorab**: Auf dem S25 ist die Inferenz **schnell** (MiDaS 2,25 ms, YOLO+MiDaS 8,5–13 ms laut README). Das Thermal-Problem ist also **nicht** primär die Inferenzzeit, sondern: (a) Kamerasensor/ISP dauerhaft auf 60–120 fps, (b) zwei unabhängige 30-fps-Loops = 60 Inference/Sek + doppelte Preprocessing desselben Frames, (c) Audio-CPU (Sinus-Regeneration 20 Hz + 500-Hz-Poll-Loop), (d) **kein** Thermal-Feedback → Throttling-Spirale, (e) Always-on Profiling/Logging im Release, (f) Colormap+Debug-Rendering jedes Frame (für Blinde unnötig), (g) FP32 statt int8 auf dem NPU, (h) managed_host-Tensor-Allokation pro Frame.

---

## 1. VERDICT-TABELLE — was sich lohnt (sortiert: Hebel ÷ Aufwand)

Legende Boost: geschätzter **Thermal-/Energie**-Beitrag (nicht FPS). E = Aufwand (L/M/H), R = Risiko (L/M/H), AI = KI-Automatisierbarkeit.

| # | Maßnahme | Verdict | Boost | E | R | AI | Code-Stellen |
|---|----------|---------|-------|---|---|----|--------------|
| 1 | **Thermal-Feedback-Loop**: `PowerManager.OnThermalStatusChangedListener` + adaptive FPS-/Auflösungs-Reduktion | ✅ TUN | hoch (verhindert Spirale) | L | L | hoch | `MainActivity.kt:174` (nur `FLAG_KEEP_SCREEN_ON`, sonst nichts); `Settings.kt` |
| 2 | **Kamera-FPS senken** `Range(60,120)`→`Range(30,30)` (oder von Settings ableiten); **Preview Use-Case entfernen**, wenn nicht sichtbar | ✅ TUN | hoch (Sensor/ISP = größter fixer Verbraucher) | L | L | hoch | `CameraManager.kt:47`, `:65-73` |
| 3 | **Audio: Sinus-Puffer cachen** (key = freq/duration/sample_rate), nur bei Settings-Änderung regenerieren | ✅ TUN | hoch (eliminiert ~108–216k Float-Trig-Ops/50 ms) | M | L | hoch | `spatial_audio.rs:373-415`; `depth_audio_source_data.rs:32-62` |
| 4 | **Audio: 2-ms-Poll-Sleep → 25–50 ms** + **lösche redundante OpenAL-Setter** (immutable Konstanten) | ✅ TUN | hoch (500 Hz → ~20–40 Hz Wakeups; 1 Kern weniger) | L | L | hoch | `spatial_audio.rs:290-310` (Sleep :307, Setter :291-299) |
| 5 | **int8-quantisiertes MiDaS als Default** (Asset existiert bereits!) | ✅ TUN | hoch (FP32→int8 auf HTP = deutliche Energieersparnis) | L | M | hoch (Qualität verifizieren) | `EyeAIApp.kt:74` vs `:78`; Asset `assets/midas_v2_1_256x256_quantized.tflite` |
| 6 | **Colormap + depthView + Debug-Preview gaten**, wenn nicht aktiviert | ✅ TUN | mittel-hoch (reine Debug-Visualisierung, Blinder sieht sie nicht) | L | L | hoch | `CameraFrameAnalyzer.kt:97-122` (`metricDepthColormap`, `depthView.setImageBitmap`); `colormap.rs`; `NativeLib.kt:91-129` |
| 7 | **Null-Frame `delay(1ms)` + Frische-Check** (kein Re-Inference auf stale Bitmap) | ✅ TUN | mittel (killt echten Busy-Spin + ~50% verschwendete Inferenz) | L | L | hoch | `CameraFrameAnalyzer.kt:130` & `:161` (else-Zweig fehlt); `:186-192` `getFrame()` |
| 8 | **managed_host-TensorBuffer reuse** (als Felder auf `LiteRtRuntime`, nicht pro Frame allokieren) | ✅ TUN | mittel (2 Allocs + 2 Host-Memcpy/Inference weg) | M | M | mittel (Interior-Mutability + unsafe-Sync-Review) | `litert_runtime.rs:185-205` |
| 9 | **RGBA→Float: NEON-Vektorisierung ODER einmal konvertieren + teilen** (derzeit 2×/Frame, ~1,43 M Float-Writes, kein NEON) | ✅ TUN | mittel | M | M | mittel (C++/JNI, numerisch identisch testen) | `NativeLib.cpp:115-123`; Caller `MetricDepthModel.kt:129`, `YoloModel.kt:68` |
| 10 | **Always-on Profiling/Logging im Release abschalten**: `scope()` cfg-gaten + `EnvFilter`/Level-Filter | ✅ TUN | mittel (jede Hot-Fn pusht `Instant::now`+SegQueue; `trace!`/`debug!` → logcat ohne Filter) | L | L | hoch | `profiling_attribute/src/lib.rs:68` (ungegated); `native_lib/src/lib.rs:181-185` (kein Filter) |
| 11 | **NPU: beide Modelle serialisieren/staffeln** (ein Shared-Executor ODER alternierende Frames) gegen HTP-Single-Slot-Contention | ✅ TUN | mittel-hoch (HTP hat 1 Graph-Slot; 2 konkurrierende Graphen → Thrash/Serialisierung) | M | M | mittel (Concurrency) | `litert_runtime.rs:99-101` (`unsafe impl Sync`); `CameraFrameAnalyzer.kt:50-61,79-163`; `native_lib/src/lib.rs:51-65` (write-Lock pro Modell) |
| 12 | **NPU-Perf-Hints + skel-lib-dir setzen** (QNN HTP burst/sustained, DispatchLibraryDir) | ⚠️ TUN-MIT-VORSICHT | ungewiss–mittel (NPU läuft schon via Default-Search; Kommentar "FIX: add needed options…") | M | H | **niedrig** (unsafe litert-sys FFI, gerätespezifisch) → **manuell** | `litert_runtime.rs:244-249`; `environment.rs:41-46` (hardcoded null); `NpuConfig.skel_library_dir` `:58-69` |
| 13 | **Vosk: kontinuierliches Listening stoppen**, wenn nicht im Gespräch (Push-to-Talk / Wakeup) | ✅ TUN | mittel (Vosk = bekannter CPU/Big-Core-Drain, default an) | M | L | hoch (UX-Change) | `VoskModel.kt:119-131`; `Settings.kt:65-68` (default true) |
| 14 | **big.LITTLE-Affinität** für Audio-/Inference-Threads (500-Hz-Thread nicht auf Big-Core) | ⚠️ OPTIONAL | mittel | M | M | niedrig (platform-spezifisch) → **manuell** | `spatial_audio.rs:107,121`; `CameraFrameAnalyzer.kt:50,57` |
| 15 | **per-Model Accelerator-Auswahl** (NPU-only für quant. MiDaS statt CPU|GPU|NPU immer) | ⚠️ OPTIONAL | mittel (vermeidet GPU-Op-Platzierung, die mit Komposition konkurriert) | M | M | mittel | `litert_runtime.rs:233-236` |

**Kumulierte Schätzung (Schritte 1–8 + 10):** realistisch **~40–55 %** weniger SoC-Last/Energie im stationären Betrieb; plus Thermal-Loop (1) verhindert die Drosselungs-Spirale, was die *nutzbare* Laufzeit überproportional verlängert (Erfahrung: 1,5–2,5×). Keine Garantie — muss mit §4 gemessen werden.

---

## 2. NICHT TUN / überbewertet (de-emphasize)

| Behauptung (z.T. aus alter Datei) | Warum NICHT / korrigiert | Beleg |
|---|---|---|
| **Bitmap/FloatBuffer-GC als aktuelle Thermal-Ursache** | Branch `fix/bitmap-floatbuf-gc` hat Ping-Pong + Reuse **bereits** umgesetzt. Verbleibender Churn ist auf LiteRT-Seite (`managed_host`), eine **andere** Schicht. | `CameraFrameAnalyzer.kt:64-70,219-229`; `NativeLib.kt:28-47` (direct ByteBuffer, zero-copy Ptr) |
| **MiDaS-Input-Normalisierung als CPU-Kosten** | `image_rgb_255_to_midas_image` wird **nie** aufgerufen — nur als Format-Label (`MiDaSImageRgb`) deklariert. Ist potentiell ein **Correctness-Bug** (un-normalisierte 0–255-Werte), **kein** Perf-Fresser. **Nicht** als Perf-Fix listen; stattdessen Qualität prüfen. | Def `tensor_buffer.rs:149`; Nutzung fehlt; `native_lib/src/lib.rs:243` nur Label; `depth_model.rs:68` nur Check |
| **"No frame dropping" / "unbounded inference"** | Falsch: `STRATEGY_KEEP_ONLY_LATEST` (queue+backpressure) wirft Frames an der Capture-Grenze; Default `DEFAULT_FRAME_RATE_LIMIT=30` drosselt. | `CameraManager.kt:51-52`; `Settings.kt:31` |
| **"Inference busy-spinnt, wenn langsamer als Cap"** | **Refuted**: dann blockiert der JNI-Inference-Call. Der **echte** Busy-Spin ist der **Null-Frame-Pfad** (kein `delay` im else). | `CameraFrameAnalyzer.kt:79-163` |
| **AHardwareBuffer/Zero-Copy für Kamera→Inference** (großer Rewrite) | Overkill: Float-FFI ist **bereits** zero-copy (Ptr auf direct ByteBuffer). Verbleibende Kopien = `managed_host`-Host-Kopien (→ #8 Reuse) + RGBA→Float-Conversion (→ #9 NEON/quant). AHB ist hochriskant (FFI/Speicher) bei marginaler Extra-Ersparnis. **Verschieben.** | `NativeLib.kt:26-47`; `NativeLib.cpp:132-159` |
| **GPU-Compute-Shader für Preprocessing** | Hochaufwand/-risiko, geringer Marginalnutzen vs. NEON (#9) / Quantisierung (#5). **Verschieben.** | — |
| **Batch-Processing mehrerer Frames** | Erhöht Latenz, braucht Modell-Mod. Für Echtzeit-Navigation kontraproduktiv. **Verschieben.** | — |

---

## 3. GOOD PATTERNS (nicht regressen!)

- **Zero-Copy Float-FFI**: `NativeFloatBuffer` = direct ByteBuffer, Ptr via `getByteBufferPtr` → `UniffiFloatBufferWrapper{ptr,len}`. Rust liest denselben Speicher. (`NativeLib.kt:26-47`; `NativeLib.cpp:132-159`)
- **Bitmap-Ping-Pong** `rotatedCameraBitmaps[2]` + `reuseRawCameraBitmap`. (`CameraFrameAnalyzer.kt:64-67,219-229`)
- **`STRATEGY_KEEP_ONLY_LATEST`** auf queue depth **und** backpressure. (`CameraManager.kt:51-52`)
- **`image.close()`** immer (auch bei null). (`CameraFrameAnalyzer.kt:232`)
- **Overlay invalidated nur bei Änderung** (`!results.contentEquals`). (`OverlayViewOD.kt:108-121`)
- **Modell-Recompile guarded**: `switchDepthModel` early-return, wenn Name+NPU unverändert. (`EyeAIApp.kt:194-202`)
- **OCR on-demand** (vom Hot-Path entfernt). (`CameraFrameAnalyzer.kt:165-184`)
- **Single-Thread-Executors pro Stage** (kein Oversubscription). (`CameraFrameAnalyzer.kt:50-61`)
- **Horner-Schema** für rel2abs-Polynom (kein `pow`). (`metric_depth_model.rs:93-100`)
- **`inferno_colormap` = 256-Entry-const-LUT** (cheap; Kosten-Problem ist, dass sie **jedes Frame** für Debug läuft → #6).
- **ByteTrack inline** auf OD-Thread (kein Extra-Thread); Output-Vec mit `with_capacity`.
- **Distance→Volume** an OpenAL `DistanceModel::LinearClamped` delegiert.
- **Audio backoff** auf 500 ms, wenn pausiert.

---

## 4. MESS-ANKER (ADB/Perfetto/Tracy) — was wo zu sehen ist

> Tracy ist **verdrahtet**, aber **nur** mit Compile-Feature `enable_tracy_profiling` (`Cargo.toml`). `ProfilingFrame::finish()` ruft `Client::running().expect(...)` → in Tracy-Build ohne Verbindung **Panic**-Risiko. Port 8086.

Pro Maßnahme das erwartete Signal:
- **#2 Kamera-FPS**: `adb shell dumpsys media.camera` / Perfetto `camera` Track → Sensor/ISP-Frequenz vor/nach.
- **#1 Thermal**: `adb shell dumpsys thermalservice` + `cat /sys/class/thermal/thermal_zone*/temp` (root). SoC-Status `THERMAL_STATUS_*` über Zeit.
- **#3/#4 Audio**: Perfetto `sched` → Tie `spatial_audio`-Thread (soll 500 Hz Wakeups zeigen → nach Fix ~20–40 Hz). Tracy-Span `process_depth_estimation_data`.
- **#5 Quant**: Tracy `run_inference` (MiDaS) vor/nach int8 — Energie via `dumpsys batterystats` Reset.
- **#6 Colormap**: Tracy `metricDepthColormap` verschwindet; `dumpsys gfxinfo` Main-Thread-Zeit sinkt.
- **#8 managed_host**: Tracy `run_inference`-Span schrumpft um Copy-Anteil; `dumpsys meminfo` Native-Heap-Wachstum flacht ab.
- **#11 HTP-Contention**: Perfetto `sched`+`freq` → zwei NPU-Threads; nach Serialisierung eine. Tracy: Depth-OD-Inferenz nicht mehr überlappend.
- **Allgemein CPU/Thread**: `adb shell top -H -p $(pidof …eyeaiapp)`; Perfetto `sched freq idle`.
- **GC/Mem**: `adb shell dumpsys meminfo <pkg>` (Native/Dalvik/GC); `am dumpheap`.
- **Logcat-Tax (#10)**: `adb logcat -s "Native Lib"` — im Release sollte nach Fix wenig/kein Output.

Quick-Baseline-Skript: siehe Antwort-Nachricht §3.

---

## 5. SELBSTKRITIK — KI vs. manuell

**KI kann sicher automatisieren** (geringes FFI/Race-Risiko):
#1 (Thermal-Listener), #2 (Kamera-FPS/Preview), #3 (Audio-Cache), #4 (Sleep+Setter), #5 (Default-Toggle), #6 (Debug-Gate), #7 (`delay`+Frische-Check), #10 (cfg-gate+Filter), #13 (Vosk-Gating), Settings-XML-Defaults.

**KI mit Vorsicht** (Review nötig):
#8 (managed_host-Reuse: Interior-Mutability `Mutex<Option<TensorBuffer>>` + Interaktion mit `unsafe impl Sync`/RwLock — lokal, aber Synchronisation prüfen), #9 (NEON: numerisch identisch, aber C++/JNI-Test nötig), #11 (Serialisieren auf Shared-Executor: Race-Potenzial, Reihenfolge/Latenz).

**Entwickler manuell** (hohes FFI/Speicher/Plattform-Risiko):
#12 (NPU-Perf-Hints/skel-dir via `unsafe` litert-sys-Symbole — gerätespezifisch, Falsch-Kompilierung bricht NPU-Init), #14 (`sched_setaffinity`-Affinität — platform-spezifisch), **`unsafe impl Sync`**-Korrektheitsreview für konkurrierenden NPU-Zugriff (Finde: `litert_runtime.rs:99-101` — Annahme "LiteRT serialisiert intern" ist **unbewiesen** für zwei *verschiedene* Modelle auf HTP), ggf. AHB-Pfad (falls je angegangen).

---

## 6. KORREKTUREN zur alten `docs/PERFORMANCE_ANALYSIS.md`

| Alter Claim | Verdict | Korrektur |
|---|---|---|
| "No frame dropping — every frame" | ❌ FALSCH | `STRATEGY_KEEP_ONLY_LATEST` wirft an Capture-Grenze; 30-fps-Cap default. |
| "Unbounded inference frequency" | ❌ FALSCH | Cap = 30 fps default (`Settings.kt:31`). Echter Spin = Null-Frame-Pfad, nicht Slow-Inference. |
| "Double copy in `litert_runtime.rs:181-205`" | ⚠️ TEILWEISE | Richtig: 2 Host-Kopien + 2 Allocs/Frame. Aber "AHardwareBuffer unused" ist kein Hebel — Float-FFI ist schon zero-copy; Kopien sind `managed_host`-seitig → Reuse (#8), nicht AHB. |
| "NativeLib.kt:71-87 ByteBuffer.allocateDirect each call" | ⚠️ TEILWEISE | Alloc nur beim ersten Mal/zub klein; Buffer wird reused. **Echte** Kosten = der per-Pixel-Loop `NativeLib.cpp:115-123` (2×/Frame). |
| "yolo_model.rs:206-220 normalization loop" | ✅ RICHTIG (aber klein) | YOLO `/255`-Loop existiert; gering vs. Rest. |
| "Tracy: app connects automatically, port 8086" | ⚠️ TEILWEISE | Tracy braucht Compile-Feature `enable_tracy_profiling`; `finish()` panic-risk ohne Host. |
| Impact-%-Schätzungen (25–30 % etc.) | ⚠️ ERFUNDEN | Nicht aus Messung abgeleitet. Siehe §1-Boost-Spalte (Mechanismus-basiert, mit Konfidenz). |
| **Verpasst**: Kamera 60–120 fps, Colormap-jedes-Frame, NPU-Perf-Hints, int8-Default, Audio-Sine-Regen+500-Hz-Poll, Vosk-continuous, kein Thermal-Listener, always-on Profiling, HTP-Single-Slot, doppelte Inferenz desselben Frames | — | Diese sind die **echten** Hebel — siehe §1. |

---

## 7. START-REIHENFOLGE (Empfehlung)

1. **#1 Thermal-Loop** + **#2 Kamera-FPS** + **#6 Debug-Gate** + **#10 Profiling-abstellen** — eine kombinierte "Atmung/Effizienz"-PR. Schnell, sicher, sofort messbar.
2. **#3 + #4 Audio** — separate PR (Rust-only). Höchster Einzel-Hebel pro Zeile.
3. **#5 int8-Default** + **#7 Null-Frame-delay** — Settings/Loop-PR.
4. **#8 managed_host-Reuse** + **#9 NEON/Dedupe** — Rust/C++-PR mit Sorgfalt.
5. **#11 NPU-Serialisierung** — Concurrency-PR, gut testen.
6. **#13 Vosk-Gating** — UX-PR (mit Team abstimmen).
7. **#12 NPU-Hints** + **#14 Affinität** — manuell, gerätespezifisch, erst wenn 1–6 ausgeschöpft.

**Nicht angehen** (vorerst): AHB-Zero-Copy, GPU-Compute-Shader, Batch-Processing, MiDaS-Normalisierung-als-Perf-Fix.
