# train_nlp_qnn_cnn_v2.py - QNN NPU-optimized without Embedding layer
import tensorflow as tf
import numpy as np
import subprocess

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

VOCAB_SIZE = 5000  # Reduziert für NPU-Kompatibilität
SEQUENCE_LENGTH = 100  # Reduziert
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

train_texts, train_labels = load_dataset("DATASET.train")
val_texts, val_labels = load_dataset("DATASET.val")

# -----------------------------
# Tokenizer vorbereiten
# -----------------------------
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH
)
vectorize_layer.adapt(train_texts)

# Tokenizer-Vokabular speichern
vocab = vectorize_layer.get_vocabulary()
with open("vocab.txt", "w", encoding="utf-8") as f:
    for token in vocab:
        f.write(token + "\n")
print("✅ Tokenizer-Vokabular gespeichert als vocab.txt")

# -----------------------------
# Texte → Integer-Sequenzen
# -----------------------------
train_sequences = vectorize_layer(np.array(train_texts))
val_sequences = vectorize_layer(np.array(val_texts))

train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels))
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# One-Hot Encoding der Token-IDs
# -----------------------------
# Konvertiere Token-IDs in One-Hot Vektoren
def preprocess_for_model(sequences):
    # One-Hot Encoding (VOCAB_SIZE x SEQUENCE_LENGTH)
    return tf.one_hot(sequences, VOCAB_SIZE, axis=-1)

# Trainingsdaten mit One-Hot
train_onehot = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels))
train_onehot = train_onehot.map(lambda x, y: (preprocess_for_model(x), y))
train_onehot = train_onehot.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_onehot = tf.data.Dataset.from_tensor_slices((val_sequences, val_labels))
val_onehot = val_onehot.map(lambda x, y: (preprocess_for_model(x), y))
val_onehot = val_onehot.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# CNN-basiertes Modell (NPU-kompatibel, ohne Embedding)
# -----------------------------
# Input ist One-Hot: (SEQUENCE_LENGTH, VOCAB_SIZE)
inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, VOCAB_SIZE), dtype=tf.float32, name="input_onehot")

# 1D Convolutions über die Vokabular-Dimension
x = tf.keras.layers.Conv1D(128, 5, activation='relu', name="conv1")(inputs)
x = tf.keras.layers.MaxPooling1D(2, name="pool1")(x)

x = tf.keras.layers.Conv1D(128, 5, activation='relu', name="conv2")(x)
x = tf.keras.layers.GlobalAveragePooling1D(name="global_pool")(x)

x = tf.keras.layers.Dense(64, activation='relu', name="dense_1")(x)
x = tf.keras.layers.Dropout(0.5, name="dropout")(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax', name="output")(x)

model = tf.keras.Model(inputs, outputs, name="nlp_classifier_cnn_onehot")

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Training
# -----------------------------
model.fit(
    train_onehot,
    epochs=5,
    validation_data=val_onehot
)

# -----------------------------
# Speichern
# -----------------------------
model.save("nlp_model_onehot_float.h5")
print("✅ Float-Modell (OneHot-CNN) gespeichert")

# -----------------------------
# TFLite-Konvertierung OHNE SELECT_TF_OPS
# -----------------------------
def representative_dataset():
    for seq in train_sequences.take(100):
        onehot = tf.one_hot(seq, VOCAB_SIZE, axis=-1).numpy().astype(np.float32)
        yield [onehot.reshape(1, SEQUENCE_LENGTH, VOCAB_SIZE)]

# Float TFLite
print("\n--- Float TFLite ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_float_model = converter.convert()
with open("nlp_model_onehot_float.tflite", "wb") as f:
    f.write(tflite_float_model)
print("✅ Float TFLite (OneHot-CNN) gespeichert")

# Dynamic Range Quantization
print("\n--- Dynamic Range Quantization ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter._experimental_disable_per_channel_quantization_for_dense_layers = True
tflite_dyn_model = converter.convert()
with open("nlp_model_onehot_dynamic.tflite", "wb") as f:
    f.write(tflite_dyn_model)
print("✅ Dynamic Range TFLite (OneHot-CNN) gespeichert")

# Float16 als Alternative
print("\n--- Float16 Quantization ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_fp16_model = converter.convert()
with open("nlp_model_onehot_float16.tflite", "wb") as f:
    f.write(tflite_fp16_model)
print("✅ Float16 TFLite (OneHot-CNN) gespeichert")

# -----------------------------
# Überprüfung auf Select/Flex Ops
# -----------------------------
print("\n" + "="*50)
print("Modelle auf Select/Flex Ops prüfen...")
print("="*50)

for model_name in ['nlp_model_onehot_float.tflite', 'nlp_model_onehot_dynamic.tflite', 'nlp_model_onehot_float16.tflite']:
    print(f'\n🔍 {model_name}:')
    try:
        interpreter = tf.lite.Interpreter(model_path=model_name)
        interpreter.allocate_tensors()

        details = interpreter.get_tensor_details()
        print(f'  ✅ Keine Fehler beim Laden')
        print(f'  📊 Tensors: {len(details)}')

        # Prüfe auf Select ops im Model binary
        with open(model_name, 'rb') as f:
            content = f.read()
            if b'SelectV2' in content:
                print(f'  ⚠️  SelectV2 Ops im Binary gefunden!')
                # Zeige wo
                idx = content.find(b'SelectV2')
                print(f'     Position: {idx}')
            else:
                print(f'  ✅ Keine SelectV2 Ops!')
            if b'Flex' in content:
                print(f'  ⚠️  Flex Ops im Binary gefunden!')
            else:
                print(f'  ✅ Keine Flex Ops!')
    except Exception as e:
        print(f'  ❌ Fehler: {e}')

print("\n✨ Alle Modelle erstellt!")
print("   Verwende nlp_model_onehot_dynamic.tflite oder nlp_model_onehot_float16.tflite für QNN NPU.")