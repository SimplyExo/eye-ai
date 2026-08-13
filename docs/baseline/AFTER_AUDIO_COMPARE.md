# Audio-Fix: Before/After Soak-Vergleich

**Fix**: Sample-Caching (Arc-geteilt, nur bei Settings-Änderung regeneriert), 3 redundante OpenAL-Setter entfernt, `set_position`-Gating, Poll 2 ms→25 ms.
**Gerät**: SM-S931B (S25). Jeweils 120 s Cool-down + 300 s Last-Soak @ 1 Hz.

## Thermale Kurve (Hauptsignal)

| Metrik | BASELINE | AFTER AUDIO | Δ |
|---|---|---|---|
| Status 0 (cool) | 0 s | **53 s** | +53 s |
| Status 1 (light) | 3 s | 55 s | — |
| Status 2 (moderate) | 127 s | 155 s | — |
| **Status 3 (severe)** | **170 s** | **37 s** | **−133 s (4,6× weniger)** |
| Time-to-status ≥1 | 1 s | 54 s | +53 s |
| Time-to-status ≥2 | 4 s | 109 s | +105 s |
| **Time-to-status ≥3** | **131 s** | **264 s** | **2× länger** |
| AP (SoC) peak | 65,3 °C (t=4 s, scharfer Spike) | **56,1 °C** (t=210 s, sanft) | **−9,2 °C** |
| AP end | 56,4 °C | 54,4 °C | −2,0 °C |
| SKIN end (absolut) | 46,6 °C | 44,7 °C | −1,9 °C |
| Battery end (absolut) | 46,5 °C | 44,3 °C | −2,2 °C |

## ⚠️ Confound — ehrliche Einschränkung

Der After-Soak startete **kälter** als die Baseline (Gerät hatte während der Builds abgekühlt):

| Start-Temp | BASELINE | AFTER |
|---|---|---|
| AP (SoC) | 48,5 °C | 47,6 °C (≈ gleich ✓) |
| SKIN | 38,2 °C | 34,6 °C (−3,6 °C kälter) |
| Battery | 38,8 °C | 31,5 °C (−7,3 °C kälter) |

→ Ein **Teil** der verzögerten Throttle-Entstehung ist der kälteren SKIN-/Batterie-Starttemperatur geschuldet, **nicht** allein dem Audio-Fix. Die SKIN-Δ (+10,1 vs +8,4) ist dadurch verfälscht — das absolute SKIN-Ende (44,7 vs 46,6) ist der faire Metric.

**Am wenigsten konfundiert (starke reale Signale):**
- **AP-Start ≈ gleich** (48,5 vs 47,6) — AP ist die SoC-Die-Temp, maßgeblich für Compute-Throttle.
- **AP-Peak 56,1 vs 65,3 (−9,2 °C)**: Die Baseline spikte bei t=4 s scharf auf 65,3 °C (vor Throttle), der Fix rampt sanft zur 56,1 °C. Eine 0,9 °C kältere Starttemperatur erklärt keinen 9,2 °C niedrigeren Peak → **echte Lastreduktion**. Der permanente Audio-Kern + die 20-Hz-Sinus-Regeneration fehlen jetzt.
- **Status-3-Dauer 37 s vs 170 s** im selben 300-s-Fenster: Selbst als das Gerät Status 3 erreichte (bei 264 s), blieb es nur 37 s dort — die Gleichgewichtstemperatur ist niedriger, das SoC kommt der Dissipation näher.

## CPU-Threads (NICHT zuverlässig über Runs vergleichbar)

Java-`Executors`-Pool-Nummern sind **nicht-deterministisch** über App-Reinstalls (`pool-6` Baseline ≠ `pool-6` After). Daher ist der Per-Thread-Vergleich nicht aussagekräftig:
- `pool-6-thread-1` 89,6%→39,2% (wahrcheinlich ein Inference-Loop, aber Nummerierung verschoben)
- Neue High-Load-Pools `pool-24` (66,6%), `pool-21` (53,5%) tauchen auf — das sind die Inference-Loops unter neuen Nummern.
- `AAudio_1` 17,2%→11,1% (leicht ↓).
- Rust-„Depth Audio"-Thread nicht in Top-N → niedrig (erwartet: 40-Hz-Poll + gecachte Samples).

→ Keine saubere CPU-Attribution möglich. Die **thermale Kurve** ist das verlässliche Signal.

## Fazit

Der Audio-Fix bringt eine **klare, materielle thermale Verbesserung** (4,6× weniger severe-Throttle-Zeit, 2× längere Time-to-severe, ~9 °C niedrigerer AP-Peak). Der Effekt ist **teilweise durch die kältere Starttemperatur konfundiert**; die am wenigsten konfundierten Signale (AP-Start ≈ gleich, AP-Peak deutlich niedriger, kurze Status-3-Phase) stützen aber eine **echte Reduktion der Dauerlast**.

**Empfehlung**: Zur Bestätigung einen Repeat-Soak mit **angleichender Starttemperatur** (Gerät auf ~Baseline-Start aufwärmen, dann kühlen auf definierten Punkt) laufen lassen. Einzelne Soaks haben Varianz.

## Artefakte
`thermal_soak_baseline.csv`, `thermal_soak_after_audio.csv`, `cpu_soak_5min_baseline.txt`, `cpu_soak_5min_after_audio.txt`, `thermal_soak_raw_{baseline,after_audio}.txt`.
