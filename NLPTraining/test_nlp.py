# infer.py
import tensorflow as tf
import numpy as np

print(tf.__version__)

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

SEQUENCE_LENGTH = 250

# -----------------------------
# Tokenizer laden
# -----------------------------
with open("vocab.txt", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]

vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=len(vocab),
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH
)
vectorize_layer.set_vocabulary(vocab)

# -----------------------------
# TFLite-Modell laden
# -----------------------------
interpreter = tf.lite.Interpreter(model_path="nlp_model_int.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Eingabe-Spezifikation:", input_details)
print("Ausgabe-Spezifikation:", output_details)

def manual_vectorization(sentence):
    f = open("vocab.txt", "r")
    lines = [i.replace("\n", "") for i in f.readlines()]

    output_array = np.zeros((1, SEQUENCE_LENGTH), dtype=np.int32)

    for i, word in enumerate(sentence.split(" ")):
        try:
            index = lines.index(word)
        except ValueError:
            index = 1
        output_array[0][i] = index

    return output_array

print(manual_vectorization("die bitte der sofort jetzt"))

# -----------------------------
# Vorhersagefunktion
# -----------------------------
def predict_tflite(sentence: str):
    seq = manual_vectorization(sentence) #vectorize_layer([sentence]).numpy().astype(np.int32)  # (1, 250)
    print(seq)
    interpreter.set_tensor(input_details[0]['index'], seq)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    return dict(zip(CLASSES, output_data))

# -----------------------------
# Testschleife
# -----------------------------
print("\nTFLite Testprogramm gestartet (Abbruch mit Ctrl+C):\n")

while True:
    text = input("Eingabe: ")
    scores = predict_tflite(text)

    for cls, score in scores.items():
        print(f"{cls:20s}: {score:.3f}")

    best_cls = max(scores, key=scores.get)
    print(f"👉 Beste Klasse: {best_cls} ({scores[best_cls]:.3f})")
    print("-" * 40)
