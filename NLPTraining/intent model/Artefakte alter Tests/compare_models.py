# compare_models.py - Modellvergleich aller drei Modelle
import tensorflow as tf
import numpy as np
import time
import os

CLASSES = [
    "TEXT_RECOGNITION",
    "OBJECT_DETECTION",
    "CHANGE_SPEECH_SPEED",
    "CHANGE_SPEAKER",
    "REDIRECT_TO_LLM",
    "OPEN_SETTINGS",
    "SET_FREQUENCY",
    "SET_BPS",
    "MEASURE_DISTANCE",
    "ABORT"
]
label_map = {name: i for i, name in enumerate(CLASSES)}

VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 250

# Modelle definieren
MODELS = {
    "Alt (LSTM)": "/home/robert/Dokumente/GitHub/eye-ai-clone/eye-ai/NLPTraining/nlp_model.tflite",
    "Neu v3 Dynamic": "nlp_model_v3_dynamic.tflite",
    "Neu v3 uint8": "nlp_model_v3_uint8.tflite",  # Wird erstellt
}

# -----------------------------
# Dataset laden
# -----------------------------
def load_dataset(filename: str):
    texts, labels = [], []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            text, label = line.strip().split(";")
            texts.append(text)
            labels.append(label_map[label])
    return texts, np.array(labels, dtype=np.float32)

print("📁 Lade Dataset...")
val_texts, val_labels = load_dataset("DATASET.val")
print(f"   {len(val_texts)} Validierungsbeispiele")

# -----------------------------
# Tokenizer laden
# -----------------------------
print("📝 Lade Tokenizer...")
with open("vocab.txt", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=len(vocab),
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH
)
vectorize_layer.set_vocabulary(vocab)

# Texte → Integer-Sequenzen
val_sequences = vectorize_layer(np.array(val_texts))

# One-Hot Encoding (für neue Modelle)
def text_to_onehot(sequences):
    return tf.one_hot(sequences, VOCAB_SIZE, axis=-1).numpy().astype(np.float32)

# Altes LSTM verwendet Strings als Input
val_texts_for_old = val_texts

# -----------------------------
# Modelle laden und evaluieren
# -----------------------------
results = {}

for model_name, model_path in MODELS.items():
    if not os.path.exists(model_path):
        print(f"⚠️  {model_name}: {model_path} nicht gefunden, übersprungen")
        continue

    print(f"\n{'='*60}")
    print(f"Lade {model_name}: {model_path}")
    print('='*60)

    try:
        # Größe
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        # TFLite laden
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"📊 Input: {input_details[0]['dtype']}, shape {input_details[0]['shape']}")
        print(f"📊 Output: {output_details[0]['dtype']}, shape {output_details[0]['shape']}")
        print(f"📊 Größe: {model_size_mb:.2f} MB")

        # Check NPU Kompatibilität
        with open(model_path, 'rb') as f:
            content = f.read()
            has_select = b'SelectV2' in content
            has_flex = b'Flex' in content
            npu_compatible = not (has_select or has_flex)
        print(f"🔌 NPU-kompatibel: {'✅' if npu_compatible else '❌'}")

        # Inference Zeit messen
        latencies = []
        num_samples = min(50, len(val_sequences))

        print(f"🚀 Messung Inferenzzeit ({num_samples} Samples)...")

        for i in range(num_samples):
            # Input vorbereiten
            if "LSTM" in model_name:
                # Altes LSTM: String Input
                input_data = np.array([val_texts_for_old[i]], dtype=np.str_)
            else:
                # Neue Modelle: One-Hot Input
                seq = val_sequences[i]
                onehot = text_to_onehot(seq[0])
                input_data = onehot

            start = time.time()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            end = time.time()

            latencies.append((end - start) * 1000)  # ms

        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)

        # Accuracy berechnen
        print(f"📈 Berechne Accuracy...")
        correct = 0
        total = len(val_sequences)

        for i in range(total):
            # Input vorbereiten
            if "LSTM" in model_name:
                input_data = np.array([val_texts_for_old[i]], dtype=np.str_)
            else:
                seq = val_sequences[i]
                onehot = text_to_onehot(seq[0])
                input_data = onehot

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            pred = np.argmax(output[0])
            true = int(val_labels[i])

            if pred == true:
                correct += 1

        accuracy = correct / total * 100

        results[model_name] = {
            "size_mb": model_size_mb,
            "npu_compatible": npu_compatible,
            "avg_latency_ms": avg_latency,
            "std_latency_ms": std_latency,
            "accuracy": accuracy,
            "input_dtype": str(input_details[0]['dtype']),
            "output_dtype": str(output_details[0]['dtype'])
        }

        print(f"\n✨ {model_name} Ergebnisse:")
        print(f"   Accuracy: {accuracy:.2f}%")
        print(f"   Latenz: {avg_latency:.2f} ± {std_latency:.2f} ms")
        print(f"   NPU-kompatibel: {'✅' if npu_compatible else '❌'}")

    except Exception as e:
        print(f"❌ Fehler bei {model_name}: {e}")
        import traceback
        traceback.print_exc()

# -----------------------------
# Zusammenfassungstabelle
# -----------------------------
print("\n" + "="*80)
print(" " * 20 + "MODELLVERGLEICH - ZUSAMMENFASSUNG")
print("="*80)

if results:
    print(f"\n{'Modell':<20} {'Größe':<10} {'Accuracy':<12} {'Latenz':<15} {'NPU':<8}")
    print("-" * 80)

    for name, res in results.items():
        npu_icon = "✅" if res['npu_compatible'] else "❌"
        print(f"{name:<20} {res['size_mb']:<10.2f} {res['accuracy']:<12.2f} {res['avg_latency_ms']:<15.2f} {npu_icon:<8}")

    print("\n" + "="*80)
    print("Empfehlung:")
    print("="*80)

    # Beste Accuracy
    best_acc = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"🏆 Beste Accuracy: {best_acc[0]} ({best_acc[1]['accuracy']:.2f}%)")

    # Beste Latenz
    best_latency = min(results.items(), key=lambda x: x[1]['avg_latency_ms'])
    print(f"🚀 Schnellste Inferenz: {best_latency[0]} ({best_latency[1]['avg_latency_ms']:.2f} ms)")

    # Beste NPU
    npu_models = {k: v for k, v in results.items() if v['npu_compatible']}
    if npu_models:
        best_npu = max(npu_models.items(), key=lambda x: x[1]['accuracy'])
        print(f"🔌 Bester NPU-kompatibel: {best_npu[0]} ({best_npu[1]['accuracy']:.2f}%)")

    print("\n---")

    # Final Empfehlung
    if npu_models:
        print("✅ Empfehlung für Snapdragon NPU (QNN):")
        best_npu_all = max(npu_models.items(), key=lambda x: (x[1]['accuracy'] / x[1]['avg_latency_ms']))
        print(f"   Verwende {best_npu_all[0]}")
        print(f"   - Accuracy: {best_npu_all[1]['accuracy']:.2f}%")
        print(f"   - Latenz: {best_npu_all[1]['avg_latency_ms']:.2f} ms")
        print(f"   - Größe: {best_npu_all[1]['size_mb']:.2f} MB")
    else:
        print("⚠️  Kein NPU-kompatibles Modell verfügbar!")