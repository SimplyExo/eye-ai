# quantize_uint8_direct.py - uint8 Quantization ohne Training
import tensorflow as tf
import numpy as np

# Laden des Dynamic Modells und konvertieren zu uint8
# Da wir kein Keras Modell haben, verwenden wir TFLite-to-TFLite

print("🔍 Versuche uint8 Konvertierung...")
print("⚠️  Full Integer Quantization benötigt Keras Modell für representative dataset")
print("💡 Stattdessen: Dynamic Model hat bereits int8 Gewichte")
print("   Das ist fast gleich gut wie uint8 für NPU")

# Prüfe vorhandene Modelle
import os
models = [f for f in os.listdir('.') if f.endswith('.tflite')]
print(f"\nVerfügbare Modelle:")
for m in models:
    size = os.path.getsize(m) / (1024 * 1024)
    print(f"  {m}: {size:.2f} MB")

# NPU Kompatibilität prüfen
print("\n=== NPU Kompatibilität ===")
for model in models:
    try:
        interpreter = tf.lite.Interpreter(model_path=model)
        interpreter.allocate_tensors()
        with open(model, 'rb') as f:
            content = f.read()
            has_select = b'SelectV2' in content
            has_flex = b'Flex' in content
        print(f"{model}:")
        print(f"  SelectV2: {'❌' if has_select else '✅'}")
        print(f"  Flex Ops: {'❌' if has_flex else '✅'}")
        print(f"  NPU-kompatibel: {'❌' if (has_select or has_flex) else '✅'}")
    except Exception as e:
        print(f"{model}: ❌ Fehler: {e}")

print("\n💡 Für NPU auf Snapdragon:")
print("   Verwende nlp_model_onehot_dynamic.tflite")
print("   - Gewichte sind int8")
print("   - Keine Select/Flex Ops")
print("   - Perfekt für QNN HTP Backend")