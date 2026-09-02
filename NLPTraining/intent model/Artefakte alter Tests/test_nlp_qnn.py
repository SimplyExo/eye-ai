# test_nlp_qnn.py - NPU-kompatible NLP Inferenz
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore')

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

VOCAB_SIZE = 5000
SEQUENCE_LENGTH = 100

# -----------------------------
# Tokenizer laden
# -----------------------------
with open("vocab.txt", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]

# -----------------------------
# Text-OneHot-Encoding
# -----------------------------
def text_to_onehot(sentence: str):
    """Konvertiert einen Text in One-Hot format für das NPU-Modell"""
    # Tokenize: Wörter zu IDs
    words = sentence.lower().split()[:SEQUENCE_LENGTH]

    # Initialisieren mit Padding-Token (ID 1)
    token_ids = np.ones(SEQUENCE_LENGTH, dtype=np.int32)

    for i, word in enumerate(words):
        try:
            token_ids[i] = vocab.index(word)
        except ValueError:
            token_ids[i] = 1  # OOV token

    # One-Hot Encoding
    onehot = np.zeros((SEQUENCE_LENGTH, VOCAB_SIZE), dtype=np.float32)
    onehot[np.arange(SEQUENCE_LENGTH), token_ids] = 1.0

    return onehot.reshape(1, SEQUENCE_LENGTH, VOCAB_SIZE)

# -----------------------------
# TFLite-Modell laden (NPU-kompatibel)
# -----------------------------
MODEL_PATH = "nlp_model_onehot_dynamic.tflite"  # Alternativ: nlp_model_onehot_float16.tflite

print(f"Lade Modell: {MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"\nEingabe-Spezifikation:")
print(f"  Shape: {input_details[0]['shape']}")
print(f"  Dtype: {input_details[0]['dtype']}")

print(f"\nAusgabe-Spezifikation:")
print(f"  Shape: {output_details[0]['shape']}")
print(f"  Dtype: {output_details[0]['dtype']}")

# -----------------------------
# Vorhersagefunktion
# -----------------------------
def predict_tflite(sentence: str):
    onehot = text_to_onehot(sentence)
    interpreter.set_tensor(input_details[0]['index'], onehot)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return dict(zip(CLASSES, output_data))

# -----------------------------
# Testschleife
# -----------------------------
print("\n" + "="*50)
print("NLP Inferenz (NPU-kompatibel, ohne Select Ops)")
print("="*50)

test_sentences = [
    "erkenne den text",
    "suche objekte",
    "änder sprechgeschwindigkeit",
    "wechsle sprecher",
    "öffne einstellungen",
    "setze frequenz",
    "miss abstand",
    "abbruch"
]

print("\nTest-Beispiele:")
for sentence in test_sentences:
    scores = predict_tflite(sentence)
    best_cls = max(scores, key=scores.get)
    print(f"  '{sentence}' -> {best_cls} ({scores[best_cls]:.3f})")

print("\n" + "="*50)
print("Interaktiver Modus (Abbruch mit Ctrl+C):")
print("="*50)

while True:
    try:
        text = input("\nEingabe: ").strip()
        if not text:
            continue

        scores = predict_tflite(text)

        for cls, score in scores.items():
            bar = "█" * int(score * 20)
            print(f"  {cls:20s}: {score:.3f} {bar}")

        best_cls = max(scores, key=scores.get)
        print(f"\n👉 Beste Klasse: {best_cls} ({scores[best_cls]:.3f})")
    except KeyboardInterrupt:
        print("\n\n👋 Beendet!")
        break
    except Exception as e:
        print(f"❌ Fehler: {e}")