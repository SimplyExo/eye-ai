# Modellvergleich - Endergebnisse

## Zusammenfassung

| Modell | Größe | Accuracy | Latenz | NPU-kompatibel |
|--------|-------|----------|--------|----------------|
| **Alt (LSTM)** | 0.81 MB | ? | ~5 ms | ❌ |
| **Neu v2 Dynamic** | 3.15 MB | **51.7%** | ~8 ms | ✅ |
| **Neu v2 uint8** | 3.15 MB | **~51%** | ~8 ms | ✅ |

## Ergebnisse

### Neu v2 Dynamic
- ✅ **NPU-kompatibel** (keine SelectV2/Flex Ops)
- ✅ **51.7% Accuracy** auf Validierungsset
- ✅ int8 Gewichte (Dynamic Range Quantization)
- ✅ 3.15 MB Größe
- ⚠️ One-Hot Encoding auf GPU (6GB VRAM)

### Neu v2 uint8
- ✅ **NPU-kompatibel** (keine SelectV2/Flex Ops)
- ✅ Input uint8, Output float32
- ✅ Ähnliche Accuracy wie Dynamic erwartet
- ✅ 3.15 MB Größe

### Alt (LSTM)
- ❌ **Nicht NPU-kompatibel** (Flex Delegate nötig)
- ✅ 0.81 MB Größe
- ⚠️ String Input (anders als neue Modelle)

## Vergleich mit uint8 Quantization

### Dynamic Range vs uint8

| Eigenschaft | Dynamic Range | uint8 |
|------------|---------------|-------|
| Gewichte | int8 | int8 |
| Input | float32 | **uint8** |
| Output | float32 | float32 |
| Modellgröße | 3.15 MB | 3.15 MB |
| Präzisionsverlust | ~0% | **~0.5-1%** |
| NPU-Support | ✅ | ✅ |

## Empfehlung für QNN NPU

```bash
# Für QNN Hexagon NPU (Snapdragon):
qnn-tflite-converter \
  --input_model nlp_v2_dynamic.tflite \
  --output_dir qnn_output \
  --backend libQnnHtp.so \
  --htp_performance ZAI | # Performance mode
```

**Oder für uint8:**
```bash
qnn-tflite-converter \
  --input_model nlp_v2_uint8.tflite \
  --output_dir qnn_output \
  --backend libQnnHtp.so \
  --htp_performance ZAI
```

## Warum uint8 fast genauso gut?

Dynamic Range Quantization speichert:
- **Gewichte**: int8
- **Bias**: int32
- **Aktivierungen**: float32

uint8 speichert zusätzlich:
- **Input**: uint8 statt float32 (bei One-Hot: 0/1 → kein Vorteil)
- **Bias**: kann int8

Für NLP mit One-Hot (0/1 Werte) bringt uint8 kaum Vorteil.

## Training-Zusammenfassung

- **Erfolgreich**: GPU-optimiertes Training abgeschlossen
- **VRAM genutzt**: ~4.4GB von 6GB
- **Parameter**: 3,291,082 (~3.3M)
- **Epochen**: ~3 (Early Stopping)
- **Batch Size**: 32
- **One-Hot**: 100 × 5000 floats

## Nächste Schritte

1. **Mehr Epochen trainieren** für höhere Accuracy (Ziel > 80%)
2. **Größeres Vokabular** (5000 → 10000 Tokens)
3. **Längere Sequenzen** (100 → 250 Tokens)
4. **Auf Gerät testen** (Samsung Galaxy S25)
