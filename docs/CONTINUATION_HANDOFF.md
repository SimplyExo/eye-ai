# EyeAI — Session-Handoff (Thermal-/Effizienz-Arbeit)

**Zweck:** Diese Datei erlaubt einer neuen Session, die Arbeit ohne Re-Analyse fortzusetzen. Sie enthält die verifizierte Analyse, die **behaltenen** Verbesserungsvorschläge (verworfene separat kurz genannt), die Messergebnisse, den aktuellen Stand und die offenen Entscheidungen.
**Letztes Update:** 2026-07-26
**Branch:** `fix/bitmap-floatbuf-gc` (Arbeitsbaum: Audio-Fix uncommitted)

---

## 1. Kontext

- **App:** EyeAI — Blinden-Navigation (Rust-Kern `eye-ai-core-rs` + Kotlin-App `EyeAIApp`, UniFFI-FFI).
- **Pipeline:** CameraX → Bitmap (JNI) → Rust (UniFFI, zero-copy Float-Ptr) → LiteRT-Inferenz (YOLOv8 + MiDaS + Rel2Abs) → Spatial-Audio (OpenAL, Parkassistent-Piepsen) + Overlay.
- **Accelerators:** LiteRT 0.2.1, QNN-HTP-NPU-Delegate + GPU-Delegate. Modelle aktuell **FP32**.
- **Problem:** Extreme Akku-/Thermal-Last → App nach kurzer Zeit unbrauchbar (drosselt).
- **Gerät zum Messen:** Samsung SM-S931B (Galaxy S25, NPU), ADB ohne Root, `dumpsys thermalservice` liefert AP/SKIN/Batterie-Sensortemperaturen.

## 2. Verifizierte Bottlenecks (Code-refs, aus 20-Agent-Workflow + first-hand)

Die Inferenz selbst ist auf dem S25 **schnell** (MiDaS 2,25 ms, YOLO+MiDaS 8,5–13 ms). Das Thermal-Problem sitzt **nicht** in der Inferenzzeit, sondern:

1. **Zwei unabhängige 30-fps-Inference-Loops** (`CameraFrameAnalyzer.kt:79-163`, depth `:79-132`, OD `:135-163`) — beide holen **dasselbe** `latestCameraFrame` (`:186-192`), je eigenes Canvas-Scale + `bitmapToRgbHwc255FloatArray` + NPU-Inferenz. = 2× Preprocessing + 2× NPU-Submissions/Frame.
2. **HTP-Single-Slot-Contention:** Beide Modelle auf NPU/HTP kompiliert (`litert_runtime.rs:233-249`), zwei residente Graphen, zwei Threads. `unsafe impl Sync` (`:99-101`) vertraut auf „LiteRT serialisiert intern" — für zwei *verschiedene* Graphen unbewiesen. „Parallel" auf Single-Slot-NPU = Fake-Parallelismus (serialisieren/thrashen).
3. **managed_host-TensorBuffer pro Frame allokiert** (`litert_runtime.rs:185-205`): pro Inferenz 2 frische `managed_host` + 2 Host-Memcpies, nicht reuse-t (im Gegensatz zur Kotlin-Seite, die der `fix/bitmap-floatbuf-gc`-Branch schon reuse-t). Symptom: +13 MB Native Heap / 35 s.
4. **RGBA→Float per-Pixel, 2×/Frame** (`NativeLib.cpp:115-123`, Caller `MetricDepthModel.kt:129`, `YoloModel.kt:68`): ~1,43 M Float-Writes/Frame, kein NEON, doppelt.
5. **ByteTrack nur Glättung, nicht Raten-Reduktion** (`object_tracker.rs`, `native_lib/src/lib.rs:400-427`): Tracker könnte OD-Rate senken, tut es nicht.
6. **NPU-Perf-Mode/skel-dir ungenutzt** (`litert_runtime.rs:246` Kommentar `// FIX: add needed options when npu is enabled!`): `NpuConfig.skel_library_dir` durchgereicht, nie an LiteRT übergeben; keine HTP-Power-Mode-Hints. NPU läuft (Stub-Libs in `jniLibs`), aber suboptimal.
7. **Audio-Subsystem** (bereits gefixt, siehe §5.8): war 2,16 M Trig-Ops/s + 500-Hz-Poll + 9× identische Buffer-Regeneration.

**Wichtige Good-Patterns (nicht regressen):** Zero-Copy-Float-FFI (`NativeLib.kt:26-47`, `NativeLib.cpp:132-159`), Bitmap-Ping-Pong (`CameraFrameAnalyzer.kt:64-70,219-229`), `STRATEGY_KEEP_ONLY_LATEST` (`CameraManager.kt:51-52`), `image.close()` immer, Overlay-invalidates-nur-bei-Änderung, Modell-Recompile-Guard, OCR on-demand, Single-Thread-Executors pro Stage, Horner-Schema für rel2abs.

## 3. Messergebnisse (S25, 5-Min-Soaks, `docs/baseline/`)

**Throttle-Treiber ist SKIN** (Schwellen 38/40/42/45/47 °C). `dumpsys thermalservice` ohne Root.

### 3.1 Baseline (Limiter-aus, kein Audio-Fix) — `thermal_soak_baseline.csv`
- Start AP 48,5 / SKIN 38,2 / bat 38,8 (Status 1).
- Status-Histogramm: 1 (3 s), 2 (127 s), **3 (170 s)**. Time-to-status-3: 131 s. Nie zurück auf 0.
- End SKIN 46,6 / bat 46,5 / AP 56,4. AP-Peak 65,3 (t=4 s, scharfer Pre-Throttle-Spike).
- CPU: zwei Inference-Threads (`pool-*` ~82 % + ~41 %), `HeapTaskDaemon` ~24 %, `CameraX` ~10 %, `AAudio` ~7 %.

### 3.2 Nach Audio-Fix, Limiter-AN (30 fps) — `thermal_soak_after_audio.csv` *(KONFUNDIERT)*
- Status 3 nur 37 s (vs 170), Time-to-3 264 s. Sah aus wie großer Gewinn.
- **ABER:** Limiter war zwischen Baseline und After eingeschaltet worden → Vergleich konfundiert (Limiter-Wechsel + Audio-Fix + Start-Temp). Der „Gewinn" war **überwiegend der 30-fps-Limiter**, nicht der Audio-Fix.

### 3.3 Matched: Audio-Fix, Limiter-AUS — `thermal_soak_after_audio_matched.csv`
- Start AP 42,9 / SKIN 39,9 / bat 41,1 (SKIN wärmer als Baseline → Confound).
- Status 3 bei 13 s, 288 s auf Status 3. **Schnelleres Throtteln = Start-Temp-Artefakt** (SKIN 39,9 vs 38,2), keine Audio-Fix-Regression.
- **Steady-state-End-Temps identisch zur Baseline:** SKIN 46,8 vs 46,6, bat 47,0 vs 46,5, AP 55,9 vs 56,4.
- **Schlussfolgerung:** Unter Real-Usage (Limiter aus) ist der Audio-Fix thermisch **vernachlässigbar** — die unbounded Inference dominiert so vollständig, dass der Audio-Kern keine Rolle spielt.

### 3.4 Kern-Mess-Erkenntnis
**Der Frame-Rate-Limiter ist der größte sofortige Thermal-Hebel** (in Real-Usage AUS; Einschalten → After-Soak zeigte dramatische Besserung). Er ist **Inferenz-Rate**, nicht Kamerasensor-Rate → kameraquellen-agnostisch (überlebt den geplanten Wechsel auf externe Kamera).

## 4. Behaltene Verbesserungsvorschläge (sortiert: Hebel ÷ Aufwand)

> Verworfene siehe §6 (nicht erneut vorschlagen).

### 4.1 Frame-Rate-Limiter aktivieren/senken *(gemessener größter Hebel, geringster Aufwand)*
- **Was:** In Real-Usage ist der Limiter AUS (`Settings.kt` `enable_*_frame_rate_limit_setting = false`). Einschalten (30 fps) oder senken (15–20 fps) drosselt die Inference-Duty massiv.
- **Code:** `Settings.kt:31` (`DEFAULT_FRAME_RATE_LIMIT=30`), `CameraFrameAnalyzer.kt:125-129,156-160` (Delay-Logik), `settings_preferences.xml`.
- **Aufwand:** L (Setting). **Risiko:** L. **KI-automatisierbar:** hoch. **Caveat:** Nur wirksam, wenn Loop-Inferenz schneller als Cap ist (sonst kein Delay → kein Effekt, siehe `busy-spin`-Korrektur: der echte Spin ist der Null-Frame-Pfad, nicht Slow-Inference).

### 4.2 Pipeline-Collapse + alternierendes Interleaving *(größter architektonischer Hebel)*
- **Was:** Zwei `while(isActive)`-Loops → **ein** Shared-Executor, **eine** gemeinsame Preprocessing-Pass, Depth (256²) und YOLO (640²) per Letterbox aus demselben Puffer, **alternierend** (Depth gerade, YOLO ungerade Frames). ~halbe NPU-Duty, null HTP-Contention, halbe Preprocessing.
- **Code:** `CameraFrameAnalyzer.kt:79-163` (beide Loops), `:186-192` (`getFrame`), `litert_runtime.rs:99-101,233-249`.
- **Aufwand:** M. **Risiko:** M (Concurrency). **KI-automatisierbar:** mittel — **Race-Condition-Stellen explizit benennen:** (a) `AtomicReference<Bitmap>`-Poller → Channel/Signal; (b) Ping-Pong-Bitmap mit 2 Konsumenten + OCR-Leser (2 Slots reichen nicht, siehe `ping-pong-bitmap-mutation-race`); (c) Shared-Executor-Scheduling-Reihenfolge; (d) `unsafe impl Sync`-Review für konkurrierenden NPU-Zugriff.

### 4.3 ByteTrack zur OD-Raten-Reduktion
- **Was:** YOLO @ 8–10 fps + Kalman-Predict zwischen Detektionen (Tracker interpoliert auf 30 fps für Overlay/Audio). ~3× weniger YOLO-NPU-Duty. Kalman-Maschinerie ist im Tracker schon da — predict-only-Schritt nach außen legen.
- **Code:** `object_tracker.rs`, `native_lib/src/lib.rs:400-427` (`runYoloOperation`), `CameraFrameAnalyzer.kt:135-163`.
- **Aufwand:** M. **Risiko:** M (Tracker-Semantik/Correctness). **KI-automatisierbar:** mittel.

### 4.4 managed_host-TensorBuffer reuse
- **Was:** Input/Output-`TensorBuffer` als Felder auf `LiteRtRuntime` cachen (Interior-Mutability `Mutex<Option<TensorBuffer>>`), nicht pro Frame allokieren. Entfernt 2 Allocs + 2 Host-Memcpies/Inference + den +13 MB/35 s Native-Heap-Churn.
- **Code:** `litert_runtime.rs:185-205` (`TensorBuffer::managed_host` je Frame), `:99-101` (`unsafe impl Sync`/RwLock-Interaktion reviewen).
- **Aufwand:** M. **Risiko:** M (Synchronisation gegen unsafe Sync). **KI-automatisierbar:** mittel.

### 4.5 RGBA→Float: NEON-Vektorisierung oder einmal konvertieren + teilen
- **Was:** Derzeit 2×/Frame (~1,43 M Float-Writes, kein NEON). Entweder NEON-vectorisieren ODER einmal konvertieren und Depth/OD teilen.
- **Code:** `NativeLib.cpp:115-123` (Loop), `:142-159` (JNI), Caller `MetricDepthModel.kt:129`, `YoloModel.kt:68`.
- **Aufwand:** M. **Risiko:** M (C++/JNI, numerisch identisch testen). **KI-automatisierbar:** mittel. **Bonus:** quantisiertes MiDaS will uint8, nicht float → RGBA→RGB-Byte-Shuffle (~3× günstiger, siehe §4.6).

### 4.6 int8-Quantisierung *(aufgeschoben, aber größter Einzel-NPU-Hebel)*
- **Was:** HTP ist int8-optimiert; FP32 kostet ~3–4× die NPU-Energie. MiDaS-quantized liegt in Assets (`EyeAIApp.kt:78`), ist nur nicht Default (`:74` fp32). YOLO-Quantisierung vom Nutzer separat geplant (vorerst ignorieren).
- **Code:** `EyeAIApp.kt:74,78`, `litert_runtime.rs:182,192` (`ElementType::Float32` hardcodiert).
- **Aufwand:** L (MiDaS-Default-Toggle) bis M (YOLO, aufgeschoben). **Risiko:** M (Qualität verifizieren). **KI-automatisierbar:** hoch (Toggle). **Status:** vom Nutzer aufgeschoben — bei Bereitschaft der größte Hebel.

### 4.7 NPU-Perf-Mode + skel-dir *(unsicher, manuell)*
- **Was:** QNN-HTP-Power-Mode (burst/sustained) + `skel_library_dir` an LiteRT übergeben (der „FIX"-Kommentar).
- **Code:** `litert_runtime.rs:244-249`, `NpuConfig` `:58-69`, `environment.rs:41-46` (null options).
- **Aufwand:** M. **Risiko:** H (`unsafe` litert-sys-FFI, gerätespezifisch). **KI-automatisierbar:** niedrig → **manuell**. **Nutzen:** ungewiss (NPU läuft schon). Priorität nach 4.1–4.5.

### 4.8 Audio-Fix *(IMPLEMENTIERT, thermisch vernachlässigbar unter Limiter-aus)*
- **Status:** Implementiert + kompilierverifiziert (`cargo check` + `clippy`, 0 W). Dateien: `eye-ai-core-rs/src/audio/{depth_audio_source_data.rs,spatial_audio.rs,math_vector.rs}`.
- **Was:** Sinus-Puffer gecacht (`Arc`-geteilt, nur bei Settings-Änderung regeneriert), 3 redundante OpenAL-Setter entfernt, `set_position`-Gating, Poll 2 ms→25 ms.
- **Wert:** Eliminiert echte Verschwendung (2,16 M Trig-Ops/s, 500-Hz-Poll), korrekt, keine Regression. **Aber:** Unter Limiter-aus thermisch vernachlässigbar (Inferenz dominiert, §3.3). Behalten — kein Pflaster, aber kein Thermal-Hebel unter Real-Usage.

## 5. Aktueller Stand

- **Audio-Fix:** implementiert, im Arbeitsbaum (uncommitted), in der installierten APK.
- **Build-Chain repariert:** `cargo-ndk` + `rustup target add aarch64-linux-android` installiert; `cargo build-android` regeneriert `.so` + UniFFI-Bindings (waren stale — pre-experimental-v2, fehlten `newDepthFrame`/`predictDepth`/`getMetricDepthModelInputShape`/`getYoloInputShape` etc.). `./gradlew assembleDebug` läuft fehlerfrei durch.
- **Baseline gemessen:** `docs/baseline/thermal_soak_baseline.csv` (Limiter-aus, kein Fix) + After-Varianten.
- **Artefakte:** `docs/baseline/{thermal_soak_baseline.csv, thermal_soak_after_audio.csv, thermal_soak_after_audio_matched.csv, cpu_soak_5min_*.txt, AFTER_AUDIO_COMPARE.md, run_thermal_soak.sh, run_baseline.sh, BASELINE_SUMMARY.md}`.
- **Detaillierte Analyse:** `docs/PERFORMANCE_ANALYSIS_VERIFIED.md` (volle Verdict-Tabelle inkl. verworfener Items + Korrekturen zur alten `docs/PERFORMANCE_ANALYSIS.md`).

## 6. Verworfene Vorschläge (nicht erneut vorschlagen)

- **Profiling/Logging im Release abstellen** (`profiling_attribute/src/lib.rs:68`, `native_lib/src/lib.rs:181-185`) — Nutzer: „Lapalie", bekannt, trivial.
- **Bildschirm aus / KEEP_SCREEN_ON-Handling** (`MainActivity.kt:174`) — Nutzer: bekannt, trivial.
- **Colormap/depthView/Debug-Preview gaten** (`CameraFrameAnalyzer.kt:97-122`, `NativeLib.kt:91-129`, `colormap.rs`) — Nutzer: bekannt, trivial (blinder Nutzer sieht's nicht).
- **Kamera-Sensor-FPS 60→30 / Preview weg** (`CameraManager.kt:47`) — kurzfristig; externe Kamera geplant → langfristig irrelevant. (Wohlunterschieden vom Frame-Rate-Limiter §4.1, der Inferenz-Rate kappt und kameraquellen-agnostisch ist.)
- **Thermal-Listener (PowerManager)** — Pflaster, drosselt nur bei Hitze statt die Basislast zu senken. (Nur als Safety-Net, nicht als Hebel.)
- **AHardwareBuffer/Zero-Copy für Kamera→Inferenz** — Float-FFI ist schon zero-copy; verbleibende Kopien sind `managed_host`-seitig (→ §4.4) + RGBA→Float (→ §4.5), nicht AHB. Hochriskant, marginal → verworfen.
- **GPU-Compute-Shader für Preprocessing** — Hochaufwand/-risiko, geringer Marginalnutzen vs. NEON (§4.5)/Quantisierung (§4.6). Verworfen.
- **MiDaS-Input-Normalisierung als Perf-Fix** — `image_rgb_255_to_midas_image` (`tensor_buffer.rs:149`) wird nie aufgerufen = potenzieller Correctness-Bug, **kein** Perf-Fresser. Nicht als Optimierung listen.

## 7. Offene Entscheidungen / nächste Schritte

1. **Audio-Fix committen?** Sauber isoliert (3 Dateien, +88/−21). Empfohlen vor dem nächsten großen Refactor.
2. **Pivot auf den größten Hebel?** Messung zeigt: Limiter (§4.1) und Pipeline-Collapse (§4.2) sind die wirklichen Hebel unter Real-Usage — nicht der Audio-Fix.
3. **Option B (saubere Audio-Fix-Attribution)?** Audio-Fix revertieren → `.so` rebuilden → Limiter-aus-Baseline auf aktuellem Stand → After. ~20–30 min. Steady-state-Signal (§3.3) deutet auf „vernachlässigbar" — nur bei Bedarf an strenger Attribution.
4. **Empfohlene Reihenfolge:** (a) Limiter-Hebel formal messen (Limiter-an-Soak, sauberer Cold-Start, als eigener Datenpunkt) → (b) Pipeline-Collapse+Interleaving (§4.2) als nächster substantieller Refactor.

## 8. Wichtige Befehle

```bash
# Android-Build (regeneriert .so + UniFFI-Bindings, braucht NDK)
export NDK_HOME=/home/robert/Android/Sdk/ndk/29.0.14206865
cd eye-ai-core-rs && cargo build-android          # → jniLibs/.so + uniffi/NativeLib.kt
cd ../EyeAIApp && ./gradlew assembleDebug           # APK

# 5-Min-Thermal-Soak (COOLDOWN env-konfigurierbar, default 120)
cd docs/baseline && bash run_thermal_soak.sh
# Vorher Baseline sichern: cp thermal_soak_5min.csv thermal_soak_<label>.csv

# Temps/Status live
adb shell "dumpsys thermalservice | grep -E 'Thermal Status:|mName=AP|mName=SKIN'; dumpsys battery | grep -m1 '^  temperature:'"
```
