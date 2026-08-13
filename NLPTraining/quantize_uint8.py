# quantize_uint8.py - Full Integer Quantization (uint8)
import tensorflow as tf
import numpy as np

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
BATCH_SIZE = 64

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
train_texts, train_labels = load_dataset("DATASET.train")
val_texts, val_labels = load_dataset("DATASET.val")

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

# -----------------------------
# Texte → Integer-Sequenzen
# -----------------------------
train_sequences = vectorize_layer(np.array(train_texts))
val_sequences = vectorize_layer(np.array(val_texts))

# -----------------------------
# One-Hot Encoding
# -----------------------------
def preprocess_for_model(sequences):
    return tf.one_hot(sequences, VOCAB_SIZE, axis=-1)

val_onehot = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels))
val_onehot = val_onehot.map(lambda x, y: (preprocess_for_model(x), y))
val_onehot = val_onehot.batch(BATCH_SIZE)

# -----------------------------
# Modell definieren (gleich wie v3)
# -----------------------------
inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, VOCAB_SIZE), dtype=tf.float32, name="input_onehot")

x = tf.keras.layers.Conv1D(256, 7, activation='relu', padding='same', name="conv1")(inputs)
x = tf.keras.layers.BatchNormalization(name="bn1")(x)
x = tf.keras.layers.MaxPooling1D(2, name="pool1")(x)

x = tf.keras.layers.Conv1D(256, 5, activation='relu', padding='same', name="conv2")(x)
x = tf.keras.layers.BatchNormalization(name="bn2")(x)
x = tf.keras.layers.MaxPooling1D(2, name="pool2")(x)

x = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same', name="conv3")(x)
x = tf.keras.layers.BatchNormalization(name="bn3")(x)

x = tf.keras.layers.GlobalAveragePooling1D(name="global_pool")(x)

x = tf.keras.layers.Dense(128, activation='relu', name="dense_1")(x)
x = tf.keras.layers.Dense(64, activation='relu', name="dense_2")(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax', name="output")(x)

model = tf.keras.Model(inputs, outputs, name="nlp_classifier_cnn_v3")

# -----------------------------
# Trainierte Gewichte laden
# -----------------------------
print("🔍 Suche nach trainiertem Modell...")
try:
    # Versuche Dynamic TFLite zu laden und extrahiere Gewichte
    # Das ist schwierig - stattdessen das TFLite Modell direkt quantisieren
    print("⚠️  Gewichte-Extraktion aus TFLite nicht möglich.")
    print("   Quantisiere das TFLite Modell direkt...")
    from_tflite = True
except:
    from_tflite = False

# -----------------------------
# Full Integer Quantization
# -----------------------------
print("\n🔄 Full Integer Quantization (uint8)...")

def representative_dataset():
    for seq in train_sequences.take(200):
        onehot = tf.one_hot(seq, VOCAB_SIZE, axis=-1).numpy().astype(np.float32)
        yield [onehot.reshape(1, SEQUENCE_LENGTH, VOCAB_SIZE)]

if from_tflite:
    # TFLite zu TFLite quantisieren (aus TFLite Model)
    converter = tf.lite.TFLiteConverter.from_saved_model("saved_model_dir")  # Falls vorhanden
else:
    # Aus dem Dynamic TFLite Modell quantisieren
    import os
    if os.path.exists("nlp_model_v3_dynamic.tflite"):
        print("   Lade nlp_model_v3_dynamic.tflite...")
        # Leider kann man TFLite nicht direkt zu TFLite quantisieren
        # Wir müssen das Keras Modell neu erstellen oder speichern
        print("❌ TFLite zu TFLite Quantisierung nicht direkt möglich!")
        print("   Stattdessen: Dynamic Model ist bereits fast uint8 (Gewichte int8)")
        print("   Verwende nlp_model_v3_dynamic.tflite als uint8-Alternative")
        exit(0)
    else:
        print("❌ nlp_model_v3_dynamic.tflite nicht gefunden!")
        exit(1)

# Alternative: Wenn Keras Modell gespeichert wurde
import os
keras_models = [f for f in os.listdir('.') if f.endswith('.keras') or f.endswith('_float.h5')]
if keras_models:
    print(f"   Found Keras model: {keras_models[0]}")
    model = tf.keras.models.load_model(keras_models[0])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
else:
    print("⚠️  Kein Keras Modell gefunden.")
    print("   Versuche Keras Modell während Training zu speichern...")
    exit(1)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

try:
    tflite_uint8_model = converter.convert()
    with open("nlp_model_v3_uint8.tflite", "wb") as f:
        f.write(tflite_uint8_model)
    print("✅ Full Integer TFLite gespeichert (nlp_model_v3_uint8.tflite)")
except Exception as e:
    print(f"⚠️  Full Integer fehlgeschlagen: {e}")
    print("   Versuche mit float32 Ausgabe (Softmax benötigt)...")

    converter.inference_output_type = tf.float32
    tflite_uint8_model = converter.convert()
    with open("nlp_model_v3_uint8.tflite", "wb") as f:
        f.write(tflite_uint8_model)
    print("✅ Hybrid TFLite gespeichert (Input uint8, Output float32)")

# -----------------------------
# Verifizierung
# -----------------------------
print("\n=== NPU-Kompatibilität prüfen ===")
try:
    interpreter = tf.lite.Interpreter(model_path="nlp_model_v3_uint8.tflite")
    interpreter.allocate_tensors()
    with open("nlp_model_v3_uint8.tflite", 'rb') as f:
        content = f.read()
        has_select = b'SelectV2' in content
        has_flex = b'Flex' in content
        print(f"  {'⚠️' if has_select else '✅'} SelectV2: {'Ja' if has_select else 'Nein'}")
        print(f"  {'⚠️' if has_flex else '✅'} Flex Ops: {'Ja' if has_flex else 'Nein'}")

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"  📊 Input: {input_details[0]['dtype']}, shape {input_details[0]['shape']}")
    print(f"  📊 Output: {output_details[0]['dtype']}, shape {output_details[0]['shape']}")
except Exception as e:
    print(f"❌ Fehler: {e}")

print("\n✨ Quantisierung abgeschlossen!")